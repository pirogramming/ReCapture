import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer


# ======================
# 설정(여기만 바꾸면 됨)
# ======================
OCR_CACHE_DIR = "ocr_cache"  # 루트 기준 폴더명. (services/ocr_cache라면 "services/ocr_cache")
OUT_JSON = "studynote_cluster_result.json"

EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 텍스트 너무 짧으면 과목 구분 불가 → 제외
MIN_TEXT_LEN = 15

# UMAP (권장)
UMAP_NEIGHBORS = 15
UMAP_COMPONENTS = 10
UMAP_METRIC = "cosine"

# HDBSCAN
MIN_CLUSTER_SIZE = 8   # 데이터 적으면 5~8
MIN_SAMPLES = 3        # 노이즈 많으면 3~5

# 과목명 힌트용 키워드 TopK
KEYWORD_TOPK = 6


# ======================
# 텍스트 추출(키 구조가 달라도 최대한 뽑기)
# ======================
def _clean_text(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"[^\w\s가-힣]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _walk_collect_strings(obj: Any, out: List[str]) -> None:
    """JSON 내부를 전부 훑으며 사람이 읽을 만한 문자열을 모음 (너무 짧은 건 제외)"""
    if obj is None:
        return
    if isinstance(obj, str):
        s = obj.strip()
        if len(s) >= 2:
            out.append(s)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _walk_collect_strings(v, out)
        return
    if isinstance(obj, list):
        for v in obj:
            _walk_collect_strings(v, out)
        return

def extract_text_from_cache_json(obj: Dict[str, Any]) -> str:
    """
    OCR 캐시 JSON에서 텍스트를 최대한 안정적으로 추출.
    - 흔한 키들(text, fullTextAnnotation, inferText 등) 우선
    - 없으면 JSON 전체에서 문자열을 긁어모아 합침(최후의 수단)
    """
    # 1) 흔한 키 우선
    candidate_keys = ["text", "ocr_text", "full_text", "fullText", "fullTextAnnotation"]
    for k in candidate_keys:
        if k in obj and isinstance(obj[k], str):
            return _clean_text(obj[k])

    # 2) Google Vision 형태 시도
    # obj["text_annotations"][0]["description"] 같은 구조가 있을 수 있음
    try:
        if "text_annotations" in obj and isinstance(obj["text_annotations"], list) and obj["text_annotations"]:
            desc = obj["text_annotations"][0].get("description")
            if isinstance(desc, str):
                return _clean_text(desc)
    except Exception:
        pass

    # 3) Clova 형태 시도: images[0].fields[].inferText
    try:
        images = obj.get("images")
        if isinstance(images, list) and images:
            fields = images[0].get("fields")
            if isinstance(fields, list) and fields:
                texts = []
                for f in fields:
                    it = f.get("inferText")
                    if isinstance(it, str) and it.strip():
                        texts.append(it.strip())
                if texts:
                    return _clean_text(" ".join(texts))
    except Exception:
        pass

    # 4) 최후: 전체 JSON에서 문자열 긁기
    bag: List[str] = []
    _walk_collect_strings(obj, bag)

    # 너무 잡음이 많아질 수 있어서, 길이 짧은 것/숫자만 같은 건 어느 정도 걸러줌
    filtered = []
    for s in bag:
        s2 = s.strip()
        if len(s2) < 2:
            continue
        if re.fullmatch(r"[\d\W_]+", s2):  # 숫자/기호만
            continue
        filtered.append(s2)

    return _clean_text(" ".join(filtered))


def guess_is_studynote(file_path: Path, obj: Dict[str, Any], text: str) -> bool:
    """
    study_note만 뽑기 위한 휴리스틱.
    - 1) 캐시 JSON 안에 category/label 비슷한 키가 있으면 그걸 우선
    - 2) 없으면 파일 경로/파일명에 study_note가 있으면 인정
    - 3) 그래도 없으면 텍스트에 '과제, 강의, 정리, theorem...' 같은 힌트가 있으면 인정(약한 기준)
    """
    # 1) category 비슷한 키
    for k in ["category", "label", "pred", "top1", "class_name", "class", "final_category"]:
        v = obj.get(k)
        if isinstance(v, str) and "study" in v.lower():
            return True
        if isinstance(v, str) and v.lower() in ["study_note", "studynote", "study"]:
            return True

    # 2) path hint
    p = str(file_path).lower()
    if "study_note" in p or "studynote" in p:
        return True

    # 3) text hint (보조)
    hints = ["강의", "과제", "정리", "증명", "정의", "정리", "theorem", "lemma", "matrix", "행렬", "고유값", "미분", "적분", "확률", "분포"]
    return any(h in text.lower() for h in hints) or any(h in text for h in hints)


def load_ocr_cache_items(cache_dir: str) -> List[Dict[str, str]]:
    base = Path(cache_dir)
    if not base.exists():
        raise FileNotFoundError(f"OCR_CACHE_DIR를 찾을 수 없음: {base.resolve()}")

    items: List[Dict[str, str]] = []
    json_files = list(base.rglob("*.json"))

    for fp in json_files:
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                continue
        except Exception:
            continue

        text = extract_text_from_cache_json(obj)
        if len(text) < MIN_TEXT_LEN:
            continue

        if not guess_is_studynote(fp, obj, text):
            continue

        # photo_id는 파일명 기반으로 일단 잡음 (필요하면 너희 이미지 PK로 교체 가능)
        photo_id = fp.stem
        items.append({"photo_id": photo_id, "text": text})

    return items


def cluster_and_save(items: List[Dict[str, str]]) -> None:
    texts = [it["text"] for it in items]

    model = SentenceTransformer(EMB_MODEL)
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype(np.float32)

    reducer = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        n_components=UMAP_COMPONENTS,
        metric=UMAP_METRIC,
        random_state=42
    )
    emb_low = reducer.fit_transform(emb)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        prediction_data=True
    )
    labels = clusterer.fit_predict(emb_low)

    # 클러스터별 과목 힌트(키워드)
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())

    cluster_keywords: Dict[int, List[str]] = {}
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        idx = np.where(labels == cid)[0]
        mean_tfidf = np.asarray(X[idx].mean(axis=0)).ravel()
        top_idx = mean_tfidf.argsort()[::-1][:KEYWORD_TOPK]
        cluster_keywords[int(cid)] = vocab[top_idx].tolist()

    out = {
        "config": {
            "ocr_cache_dir": OCR_CACHE_DIR,
            "emb_model": EMB_MODEL,
            "min_text_len": MIN_TEXT_LEN,
            "umap": {
                "neighbors": UMAP_NEIGHBORS,
                "components": UMAP_COMPONENTS,
                "metric": UMAP_METRIC,
            },
            "hdbscan": {
                "min_cluster_size": MIN_CLUSTER_SIZE,
                "min_samples": MIN_SAMPLES,
            }
        },
        "cluster_keywords": cluster_keywords,
        "assignments": [
            {"photo_id": it["photo_id"], "cluster_id": int(lb)}
            for it, lb in zip(items, labels)
        ]
    }

    Path(OUT_JSON).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip([int(u) for u in unique], [int(c) for c in counts]))
    print("✅ 샘플 수:", len(items))
    print("📌 클러스터 분포:", dist)
    print("📌 cluster_keywords(과목 힌트):")
    for cid, kws in cluster_keywords.items():
        print(f"  - {cid}: {kws}")
    print(f"\n✅ 저장 완료: {OUT_JSON}")


def main():
    items = load_ocr_cache_items(OCR_CACHE_DIR)
    if len(items) < 10:
        raise RuntimeError(
            f"study_note로 판정된 데이터가 너무 적어요: {len(items)}개\n"
            f"- OCR_CACHE_DIR 경로가 맞는지 확인\n"
            f"- 캐시 JSON에 category가 없으면 guess_is_studynote가 놓칠 수 있음(그럼 조건 완화 필요)"
        )
    cluster_and_save(items)

if __name__ == "__main__":
    main()
