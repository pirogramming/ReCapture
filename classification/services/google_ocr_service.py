from google.cloud import vision
import os
import io
from typing import Tuple

# ✅ HEIC 지원
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()


class GoogleOCRService:
    """Google Cloud Vision API OCR 서비스 (HEIC 지원 + 캐시 호환 유지)"""

    def __init__(self):
        print("🔧 Google Cloud Vision API 초기화 중...")

        current_dir = os.path.dirname(__file__)  # services 폴더
        parent_dir = os.path.dirname(current_dir)  # classification 폴더
        credentials_path = os.path.join(parent_dir, "google_credentials.json")

        if not os.path.exists(credentials_path):
            raise FileNotFoundError(
                f"❌ google_credentials.json 파일이 없습니다!\n"
                f"예상 경로: {credentials_path}\n"
                f"STEP 6을 다시 확인하세요."
            )

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        try:
            self.client = vision.ImageAnnotatorClient()
            print("✅ Google Cloud Vision API 준비 완료!\n")
        except Exception as e:
            raise Exception(f"❌ Vision API 초기화 실패: {e}")

    def _read_image_bytes_for_vision(self, image_path: str) -> bytes:
        """
        Vision API 입력용 bytes 생성.
        - jpg/png/webp 등: 그대로 bytes
        - heic/heif: PIL로 열어서 JPEG로 인코딩한 bytes (Vision 호환 안정)
        """
        ext = os.path.splitext(image_path)[1].lower()

        if ext in (".heic", ".heif"):
            # ✅ HEIC -> RGB -> JPEG bytes
            img = Image.open(image_path).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95, optimize=True)
            return buf.getvalue()

        # 기본: 원본 그대로
        with io.open(image_path, "rb") as f:
            return f.read()

    def extract_text(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ 이미지를 찾을 수 없습니다: {image_path}")

        print(f"📸 이미지 분석 중: {os.path.basename(image_path)}")

        content = self._read_image_bytes_for_vision(image_path)
        image = vision.Image(content=content)

        response = self.client.document_text_detection(
            image=image,
            image_context={"language_hints": ["ko", "en"]},
        )

        if response.error.message:
            raise Exception(f"❌ Vision API 오류: {response.error.message}")

        if not response.full_text_annotation:
            print("⚠️ 텍스트를 찾을 수 없습니다.")
            return {"full_text": "", "lines": [], "confidences": []}

        full_text = response.full_text_annotation.text

        lines = []
        confidences = []
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                block_text = ""
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_text = "".join([symbol.text for symbol in word.symbols])
                        block_text += word_text + " "
                if block_text.strip():
                    lines.append(block_text.strip())
                    confidences.append(block.confidence)

        print(f"✅ 텍스트 추출 완료 ({len(lines)}개 블록)")

        return {"full_text": full_text, "lines": lines, "confidences": confidences}
