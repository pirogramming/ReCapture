# classification/training/cluster_studynote_hybrid_hash.py
# (HEIC 포함) 열 수 없는 이미지는 자동 스킵 + pillow-heif opener 등록 버전

import json, re, hashlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from PIL import UnidentifiedImageError

# ✅ HEIC/HEIF 열기 지원(이미 pillow-heif 설치돼있음)
from pillow_heif import register_heif_opener
register_heif_opener()

from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

import open_clip


# ======================
# 설정
# ======================
OCR_CACHE_DIR = "classification/ocr_cache"
IMAGE_DIR = "classification/train_data/study_note"   # 필요시 test_data로 바꿔도 됨
OUT_JSON = "studynote_cluster_hybrid_hash.json"

TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

MIN_TEXT_LEN = 15
TEXT_WEIGHT = 1.0
IMAGE_WEIGHT = 1.0

UMAP_NEIGHBORS = 15
UMAP_COMPONENTS = 10
MIN_CLUSTER_SIZE = 8
MIN_SAMPLES = 3
KEYWORD_TOPK = 6


# ======================
# OCR JSON -> 텍스트 추출
# ======================
def clean_text(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def walk_collect_strings(obj: Any, out: List[str]) -> None:
    if obj is None:
        return
    if isinstance(obj, str):
        s = obj.strip()
        if len(s) >= 2:
            out.append(s)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            walk_collect_strings(v, out)
        return
    if isinstance(obj, list):
        for v in obj:
            walk_collect_strings(v, out)
        return

def extract_text(obj: Dict[str, Any]) -> str:
    v = obj.get("full_text")
    if isinstance(v, str) and v.strip():
        return clean_text(v)

    lines = obj.get("lines")
    if isinstance(lines, list):
        lines = [str(x).strip() for x in lines if str(x).strip()]
        if lines:
            return clean_text(" ".join(lines))

    bag: List[str] = []
    walk_collect_strings(obj, bag)
    return clean_text(" ".join(bag))


# ======================
# 이미지 해시 -> 캐시 매칭
# ======================
def sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def list_images(image_dir: str) -> List[Path]:
    exts = {".jpg",".jpeg",".png",".webp",".bmp",".heic",".heif"}
    out = []
    for fp in Path(image_dir).rglob("*"):
        if fp.is_file() and fp.suffix.lower() in exts:
            out.append(fp)
    return out

def load_pil_image(path: Path):
    """열 수 없는 이미지는 None 반환(스킵)"""
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        print(f"⚠️ 열 수 없는 이미지 제외: {path} ({type(e).__name__})")
        return None


def main():
    cache_base = Path(OCR_CACHE_DIR)

    img_paths = list_images(IMAGE_DIR)
    if not img_paths:
        raise RuntimeError(f"이미지가 없어요: {IMAGE_DIR}")
    print(f"✅ 이미지 후보: {len(img_paths)}")

    # 1) sha256로 ocr_cache 매칭 + 텍스트 추출
    items = []
    miss = 0

    for img in img_paths:
        h = sha256_hex(img)
        j = cache_base / f"{h}.json"
        if not j.exists():
            miss += 1
            continue

        try:
            obj = json.loads(j.read_text(encoding="utf-8"))
        except Exception:
            miss += 1
            continue

        if not isinstance(obj, dict):
            miss += 1
            continue

        text = extract_text(obj)
        if len(text) < MIN_TEXT_LEN:
            continue

        items.append({"photo_id": img.stem, "hash": h, "img_path": str(img), "text": text})

    print(f"✅ 매칭 성공: {len(items)} / 실패: {miss}")
    if len(items) < 10:
        raise RuntimeError("매칭된 샘플이 너무 적어요. (ocr_cache가 충분히 쌓였는지 확인)")

    # 2) 텍스트 임베딩
    texts = [it["text"] for it in items]
    text_model = SentenceTransformer(TEXT_MODEL)
    text_emb = text_model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    text_emb = np.asarray(text_emb, dtype=np.float32) * float(TEXT_WEIGHT)

    # 3) 이미지 임베딩(CLIP) — 이미지 못 열면 해당 샘플 제외
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    clip_model = clip_model.to(device)
    clip_model.eval()

    img_embs = []
    kept_items = []
    kept_text_emb = []

    with torch.no_grad():
        for it, te in zip(items, text_emb):
            img = load_pil_image(Path(it["img_path"]))
            if img is None:
                continue

            x = preprocess(img).unsqueeze(0).to(device)
            feat = clip_model.encode_image(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)

            img_embs.append(feat.cpu().numpy()[0])
            kept_items.append(it)
            kept_text_emb.append(te)

    items = kept_items
    text_emb = np.asarray(kept_text_emb, dtype=np.float32)
    img_emb = np.asarray(img_embs, dtype=np.float32) * float(IMAGE_WEIGHT)

    print(f"✅ 최종 사용 샘플(이미지 로드 성공): {len(items)}")
    if len(items) < 10:
        raise RuntimeError("이미지 로드 성공 샘플이 너무 적어요. (HEIC 파일들이 전부 실패하는지 확인)")

    # 4) concat + scale
    X = np.concatenate([text_emb, img_emb], axis=1)
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X).astype(np.float32)

    # 5) UMAP + HDBSCAN
    reducer = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        n_components=UMAP_COMPONENTS,
        metric="euclidean",
        random_state=42
    )
    X_low = reducer.fit_transform(X)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        prediction_data=True
    )
    labels = clusterer.fit_predict(X_low)

    # 6) 클러스터 키워드(텍스트 힌트) — 최종 items 기준으로 다시 texts 구성
    texts = [it["text"] for it in items]
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    Tf = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())

    cluster_keywords: Dict[int, List[str]] = {}
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        idx = np.where(labels == cid)[0]
        mean_tfidf = np.asarray(Tf[idx].mean(axis=0)).ravel()
        top_idx = mean_tfidf.argsort()[::-1][:KEYWORD_TOPK]
        cluster_keywords[int(cid)] = vocab[top_idx].tolist()

    out = {
        "config": {
            "ocr_cache_dir": OCR_CACHE_DIR,
            "image_dir": IMAGE_DIR,
            "text_model": TEXT_MODEL,
            "clip_model": f"{CLIP_MODEL}/{CLIP_PRETRAINED}",
            "weights": {"text": TEXT_WEIGHT, "image": IMAGE_WEIGHT},
            "umap": {"neighbors": UMAP_NEIGHBORS, "components": UMAP_COMPONENTS},
            "hdbscan": {"min_cluster_size": MIN_CLUSTER_SIZE, "min_samples": MIN_SAMPLES},
        },
        "cluster_keywords": cluster_keywords,
        "assignments": [
            {"photo_id": it["photo_id"], "hash": it["hash"], "cluster_id": int(lb)}
            for it, lb in zip(items, labels)
        ]
    }

    Path(OUT_JSON).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip([int(u) for u in unique], [int(c) for c in counts]))
    print("📌 클러스터 분포:", dist)
    print("📌 cluster_keywords:")
    for cid, kws in cluster_keywords.items():
        print(f"  - {cid}: {kws}")
    print(f"✅ 저장 완료: {OUT_JSON}")


if __name__ == "__main__":
    main()
