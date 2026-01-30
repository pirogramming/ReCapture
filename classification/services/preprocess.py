# classification/services/preprocess.py
from __future__ import annotations
from PIL import Image, ImageEnhance, ImageFilter
import os
import uuid

def preprocess_for_ocr(
    image_path: str,
    out_dir: str,
    *,
    max_side: int = 1800,
    contrast: float = 1.5,
    sharpness: float = 1.3,
    to_grayscale: bool = True,
) -> str:
    """
    OCR을 위한 전처리 이미지 파일을 생성하고 그 경로를 반환.
    - max_side: 긴 변 기준 리사이즈 (속도/안정성)
    - contrast/sharpness: 문서/영수증에 도움이 됨
    """
    os.makedirs(out_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    long_side = max(w, h)

    # 1) 리사이즈+압축(속도개선)
    if long_side > max_side:
        scale = max_side / long_side
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    # 2) 그레이스케일+대비향상(정확도개선)
    if to_grayscale:
        img = img.convert("L")  # grayscale

    # 3) 샤프닝(글자 경계 강화)
    if contrast and contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)

    # 4) 후처리(OCR결과 텍스트처리)
    if sharpness and sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
        # 약하게 한 번 더 경계 강화(선택)
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))

    # 저장
    out_name = f"ocr_{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(out_dir, out_name)

    # 품질은 적당히 줄여 속도/용량 최적화
    img.save(out_path, format="JPEG", quality=85, optimize=True)

    return out_path
