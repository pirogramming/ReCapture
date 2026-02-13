# test_clustering.py
import os
import glob

from services.ensemble_classifier import EnsembleClassifier
from services.clustering_service import ClusteringService, ClusterItem

# ✅ train_data의 study_note로 테스트
TEST_ROOT = "train_data/study_note"
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif")


def collect_images(root_dir: str):
    """
    - root_dir 바로 아래 파일만 수집
    - 확장자 대/소문자 섞여도 중복 없이 수집
    """
    patterns = []
    for ext in EXTS:
        patterns.append(os.path.join(root_dir, f"*{ext}"))
        patterns.append(os.path.join(root_dir, f"*{ext.upper()}"))

    seen = set()
    files = []
    for pat in patterns:
        for p in glob.glob(pat):
            # ✅ 절대경로+정규화해서 중복 제거
            ap = os.path.normcase(os.path.abspath(p))
            if ap in seen:
                continue
            seen.add(ap)
            files.append(p)

    files.sort(key=lambda x: os.path.basename(x).lower())
    return files


def main():
    abs_root = os.path.abspath(TEST_ROOT)
    print("TEST_ROOT =", abs_root)

    if not os.path.isdir(TEST_ROOT):
        print(f"⚠️ 폴더 없음: {TEST_ROOT}")
        return

    files = collect_images(TEST_ROOT)
    print(f"📌 study_note 후보 이미지: {len(files)}장")

    # 1) 1차 분류 + OCR 텍스트 확보 (기존 ensemble 로직 사용)
    clf = EnsembleClassifier()

    items = []
    for p in files:
        out = clf.classify(p)

        # ✅ 'study_note'로 분류된 것만 2차(클러스터링) 입력으로 사용
        if out.get("category") != "study_note":
            continue

        # ⚠️ 지금은 preview(100자)만 사용 중
        # 군집 품질 올리려면 ensemble에서 ocr_full_text를 리턴하게 바꿔서 여기서 쓰는 걸 추천
        text = (out.get("ocr_text_preview") or "").strip()
        items.append(ClusterItem(path=p, text=text))

    print(f"✅ study_note로 확정된 이미지: {len(items)}장")

    # 2) HDBSCAN 클러스터링
    clustering = ClusteringService(
        min_cluster_size=3,   # 데이터 적으면 2~3 추천
        min_samples=None,
        topk_keywords=5,
    )
    result = clustering.cluster_items(items)

    print("\n==============================")
    print("📌 CLUSTERING RESULT")
    print("==============================")
    print(result["summary"])

    print("\n[clusters]")
    for cid, info in result["clusters"].items():
        print(f"- cluster {cid}: {len(info['images'])} images")
        print(f"  keywords: {info['keywords']}")
        print(f"  suggested_name: {info['suggested_name']}")
        for x in info["images"][:3]:
            print(f"    - {os.path.basename(x)}")

    print("\n[noise]")
    for x in result["noise"][:5]:
        print(f"  - {os.path.basename(x)}")
    if len(result["noise"]) > 5:
        print(f"  ... {len(result['noise'])-5} more")


if __name__ == "__main__":
    main()
