import os
from classification.services.preprocess import preprocess_for_ocr
from classification.services.ocr_service import extract_text

def main():
    img_path = "test_images/문서정보_1.JPG"  # 여기에 네 테스트 이미지 넣기
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"이미지 파일이 없어요: {img_path}")

    pre_path = preprocess_for_ocr(img_path, out_dir="tmp_ocr")
    print("[OK] preprocessed:", pre_path)

    text = extract_text(pre_path)
    print("\n========== OCR RESULT ==========")
    print(text)
    print("================================")

if __name__ == "__main__":
    main()
