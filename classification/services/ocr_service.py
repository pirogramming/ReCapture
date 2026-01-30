import re
from paddleocr import PaddleOCR

_ocr = None

def get_ocr():
    global _ocr
    if _ocr is None:
        # 2.x 방식: 한국어 + 각도분류기(기울기 보정) 켬
        _ocr = PaddleOCR(
            lang="korean",
            use_angle_cls=True,
            show_log=False,   # 로그 너무 많으면 False 추천
        )
    return _ocr

def extract_text(image_path: str) -> str:
    ocr = get_ocr()

    # 2.x 방식: cls=True 가능 (각도분류기 적용)
    result = ocr.ocr(image_path, cls=True)

    if not result:
        return ""

    lines = []
    # result는 보통 "페이지 리스트" 구조인데 단일 이미지면 result가 바로 라인 리스트일 때도 있어서 방어
    page = result[0] if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list) else result

    for line in page:
        # line: [box, (text, score)]
        try:
            txt = line[1][0]
        except Exception:
            continue
        if txt and str(txt).strip():
            lines.append(str(txt).strip())

    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text
