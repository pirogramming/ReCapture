# classification/training/build_ocr_cache_train2.py
# ✅ classification/train_2 이미지들만 OCR 돌려서 sha256.json 캐시 생성
# ✅ 캐시 파일명 = 이미지 파일 바이트 sha256 (하이브리드 클러스터링과 동일 규칙)
# ✅ 이미 캐시 있으면 스킵(중복 과금 방지)
# ✅ HEIC/HEIF 포함(pillow-heif)
#
# ⚠️ 실행 위치 주의
# 1) 프로젝트 루트에서 실행(권장):
#    python classification/training/build_ocr_cache_train2.py
# 2) classification 폴더 안에서 실행해도 되게 sys.path 보정 포함

import json
import hashlib
import sys
from pathlib import Path
from typing import Callable, List, Optional

from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener

register_heif_opener()

# ✅ 어디서 실행하든 import가 되도록 "프로젝트 루트"를 sys.path에 추가
# - 이 파일: .../classification/training/build_ocr_cache_train2.py
# - parents[2] = 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ✅ 이제 루트 기준으로 classification 패키지 import 가능
from classification.services.google_ocr_service import GoogleOCRService  # noqa: E402


# ======================
# 설정(여기만 바꾸면 됨)
# ======================
IMAGE_DIR = "classification/train_2"
OCR_CACHE_DIR = "classification/ocr_cache"

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}


# ======================
# 유틸
# ======================
def sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_images(image_dir: str) -> List[Path]:
    base = Path(image_dir)
    return [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]


def can_open_image(path: Path) -> bool:
    try:
        Image.open(path).convert("RGB")
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def pick_ocr_method(ocr: GoogleOCRService) -> Callable[[str], object]:
    """
    GoogleOCRService에 어떤 메서드명이 있는지 환경마다 달라서 자동 선택.
    아래 후보 중 존재하는 걸 먼저 사용.
    """
    candidates = [
        "detect_text",
        "run_ocr",
        "ocr_image",
        "extract_text",
        "process_image",
        "predict",
    ]
    for name in candidates:
        fn = getattr(ocr, name, None)
        if callable(fn):
            return fn
    raise RuntimeError(
        "GoogleOCRService에서 OCR 호출 메서드를 찾지 못했어요.\n"
        "services/google_ocr_service.py에서 OCR 실행 함수명을 확인해줘."
    )


def normalize_payload(result) -> dict:
    """
    클러스터링 코드가 기대하는 포맷:
      - full_text (str)
      - lines (list)
      - confidences (list)
    result가 dict이고 full_text가 있으면 그대로 사용, 아니면 full_text만이라도 저장.
    """
    if isinstance(result, dict):
        if "full_text" in result:
            result.setdefault("lines", [])
            result.setdefault("confidences", [])
            return result
        # dict지만 포맷 다르면 json 문자열로 저장
        return {"full_text": json.dumps(result, ensure_ascii=False), "lines": [], "confidences": []}

    return {"full_text": str(result), "lines": [], "confidences": []}


# ======================
# 메인
# ======================
def main():
    cache_dir = Path(OCR_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(IMAGE_DIR)
    if not images:
        raise RuntimeError(f"이미지가 없어요: {IMAGE_DIR}")

    ocr = GoogleOCRService()
    ocr_fn = pick_ocr_method(ocr)

    total = len(images)
    new_cache = 0
    skipped = 0
    failed = 0

    print(f"✅ 대상 이미지: {total} (from {IMAGE_DIR})")
    print(f"✅ 사용 OCR 메서드: {ocr_fn.__name__}")
    print(f"📁 캐시 폴더: {OCR_CACHE_DIR}\n")

    for idx, img in enumerate(images, 1):
        h = sha256_hex(img)
        out_path = cache_dir / f"{h}.json"

        # ✅ 이미 캐시 있으면 스킵(중복 과금 방지)
        if out_path.exists():
            skipped += 1
            continue

        # ✅ 열 수 없는 파일은 스킵
        if not can_open_image(img):
            print(f"⚠️ 열 수 없는 이미지 스킵: {img}")
            failed += 1
            continue

        try:
            result = ocr_fn(str(img))
            payload = normalize_payload(result)

            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            new_cache += 1

        except Exception as e:
            print(f"❌ OCR 실패: {img} ({type(e).__name__}: {e})")
            failed += 1

        if idx % 10 == 0 or idx == total:
            print(f"Progress: {idx}/{total} | new_cache={new_cache} | skipped={skipped} | failed={failed}")

    print("\n✅ 완료!")
    print(f" - 새로 생성된 캐시: {new_cache}")
    print(f" - 기존 캐시라 스킵: {skipped}")
    print(f" - 실패/스킵: {failed}")
    print(f"📁 캐시 폴더: {OCR_CACHE_DIR}")


if __name__ == "__main__":
    main()
