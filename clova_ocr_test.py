import os
import json
import uuid
import time
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

INVOKE_URL = os.getenv("CLOVA_OCR_INVOKE_URL")
SECRET = os.getenv("CLOVA_OCR_SECRET")

if not INVOKE_URL or not SECRET:
    raise RuntimeError("❌ .env에 CLOVA_OCR_INVOKE_URL / CLOVA_OCR_SECRET를 설정해줘!")


def img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def normalize_format(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lstrip(".").lower()
    if ext == "jpeg":
        ext = "jpg"
    if ext not in ("jpg", "png", "bmp", "tiff", "pdf"):
        # 보통 jpg/png가 가장 안정적
        raise ValueError(f"지원하지 않는 확장자: {ext} (jpg/png 권장)")
    return ext


def call_clova_ocr(image_path: str) -> dict:
    payload = {
        "version": "V2",
        "requestId": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
        "images": [
            {
                "name": os.path.basename(image_path),
                "format": normalize_format(image_path),
                "data": img_to_base64(image_path),
            }
        ],
    }

    headers = {
        "X-OCR-SECRET": SECRET,
        "Content-Type": "application/json",
    }

    r = requests.post(INVOKE_URL, headers=headers, data=json.dumps(payload))
    print("HTTP status:", r.status_code)

    # 에러일 때 메시지 확인하기 쉽게
    if r.status_code != 200:
        try:
            print("Error body:", r.json())
        except Exception:
            print("Error text:", r.text)
        r.raise_for_status()

    return r.json()


def extract_tokens(result: dict) -> list[str]:
    """
    CLOVA OCR 결과에서 인식된 텍스트 토큰(inferText)만 뽑아서 리스트로 반환
    """
    images = result.get("images", [])
    if not images:
        return []
    fields = images[0].get("fields", [])
    tokens = []
    for f in fields:
        t = (f.get("inferText") or "").strip()
        if t:
            tokens.append(t)
    return tokens


if __name__ == "__main__":
    # ✅ 여기에 테스트 이미지 경로 지정
    # 같은 폴더에 sample.jpg 넣으면 "sample.jpg"로 OK
    test_image = "test_images/문서정보_7.jpg"

    out = call_clova_ocr(test_image)

    print("\n--- RAW JSON (truncated view) ---")
    print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])  # 너무 길면 앞부분만

    tokens = extract_tokens(out)
    print("\n--- EXTRACTED TEXT ---")
    print(" ".join(tokens))
