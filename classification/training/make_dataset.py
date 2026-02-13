import os
import pandas as pd
import sys

# OCR 텍스트 데이터셋 만들기 -> train_text.csv 생성

# -----------------------------------------------------------
# 1. 경로 설정
# -----------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -----------------------------------------------------------
# 2. 임포트
# -----------------------------------------------------------
from services.google_ocr_service import GoogleOCRService
from services.ocr_cache import OCRCache

# -----------------------------------------------------------
# 3. 허용 카테고리 (폴더명 기준) ✅ 4개로 수정
# -----------------------------------------------------------
VALID_CATEGORIES = [
    "finance",
    "study_note",
    "info",
    "others",
]


# -----------------------------------------------------------
# 4. CSV 생성 함수
# -----------------------------------------------------------
def create_dataset_csv():
    data_dir = "train_data"
    output_file = "train_text.csv"
    cache_dir = "ocr_cache"

    # OCR 서비스 생성
    try:
        ocr = GoogleOCRService()
    except Exception as e:
        print(f"\n❌ OCR 서비스 초기화 실패: {e}")
        print("👉 GOOGLE_APPLICATION_CREDENTIALS 환경변수를 확인하세요.")
        return

    # 캐시 생성
    cache = OCRCache(cache_dir=cache_dir)

    results = []

    print("=" * 60)
    print("📝 구글 OCR로 텍스트 추출 시작")
    print("✅ 캐시 사용: 같은 이미지는 OCR 재호출 없음")
    print(f"✅ 허용 폴더(4개): {VALID_CATEGORIES}")
    print("=" * 60)

    # -----------------------------------------------------------
    # train_data 존재 여부 확인
    # -----------------------------------------------------------
    if not os.path.exists(data_dir):
        print(f"❌ '{data_dir}' 폴더가 없습니다.")
        print("👉 먼저 train_data 폴더를 만들고 카테고리 폴더를 생성하세요.")
        return

    # 카테고리 폴더 검사
    folders = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    # 잘못된 폴더 체크
    unknown = [c for c in folders if c not in VALID_CATEGORIES]
    if unknown:
        print(f"❌ train_data 안에 알 수 없는 폴더가 있어요: {unknown}")
        print(f"✅ 허용 폴더(4개): {VALID_CATEGORIES}")
        return

    # 빠진 폴더 체크 (경고만)
    missing = [c for c in VALID_CATEGORIES if c not in folders]
    if missing:
        print(f"⚠️ 경고: 다음 카테고리 폴더가 없습니다: {missing}")

    # -----------------------------------------------------------
    # 이미지 확장자 정의
    # -----------------------------------------------------------
    exts = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".tif", ".tiff")

    # 총 이미지 수 계산
    total_files = 0
    for r, d, files in os.walk(data_dir):
        total_files += sum(1 for f in files if f.lower().endswith(exts))

    processed_count = 0
    cache_hit = 0
    cache_miss = 0

    print(f"📂 총 {total_files}장의 이미지를 처리합니다...\n")

    # -----------------------------------------------------------
    # 카테고리별 처리
    # -----------------------------------------------------------
    for category in folders:
        folder_path = os.path.join(data_dir, category)

        print(f"📂 [{category}] 폴더 처리 중...")

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(exts)]
        files = sorted(files)

        for filename in files:
            img_path = os.path.join(folder_path, filename)

            try:
                # ===== 캐시 확인 =====
                cached = cache.load(img_path)

                if cached is not None:
                    print(f"   ⚡ HIT  {category}/{filename}")
                    response = cached
                    cache_hit += 1
                else:
                    print(f"   🧠 MISS {category}/{filename} (OCR 호출)")
                    response = ocr.extract_text(img_path)
                    cache.save(img_path, response)
                    cache_miss += 1

                # ===== 텍스트 추출 =====
                if isinstance(response, dict):
                    text = response.get("full_text", "")
                else:
                    text = str(response)

                # ===== CSV 저장용 =====
                results.append({
                    "filename": filename,
                    "category": category,
                    "text": text
                })

                processed_count += 1

                if processed_count % 5 == 0:
                    print(
                        f"   👉 {processed_count}/{total_files} 완료 "
                        f"(HIT={cache_hit}, MISS={cache_miss})"
                    )

            except Exception as e:
                print(f"   ⚠️ 에러 발생 ({filename}): {e}")

    # -----------------------------------------------------------
    # CSV 저장
    # -----------------------------------------------------------
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 60)
        print(f"🎉 변환 완료! '{output_file}' 생성")
        print(f"   총 데이터: {len(results)}")
        print(f"   ✅ 캐시 히트: {cache_hit}")
        print(f"   ✅ 캐시 미스: {cache_miss}")
        print(f"   📁 캐시 위치: {os.path.abspath(cache_dir)}")
        print("   📌 카테고리:", sorted(df["category"].unique()))
        print("=" * 60)

        print("미리보기:")
        print(df.head())

    else:
        print("\n❌ 저장할 데이터가 없습니다.")


# -----------------------------------------------------------
# 실행
# -----------------------------------------------------------
if __name__ == "__main__":
    create_dataset_csv()
