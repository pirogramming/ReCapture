import os
import pandas as pd
from collections import Counter

print("="*60)
print("🔍 누락 이미지 검사 시작")
print("="*60)

# -----------------------------
# 1️⃣ CSV 로드
# -----------------------------
df = pd.read_csv("train_text.csv")

# CSV에 저장된 경로 (category/filename)
csv_paths = set(
    df["category"] + "/" + df["filename"]
)

# -----------------------------
# 2️⃣ train_data 전체 스캔
# -----------------------------
exts = ('.jpg', '.jpeg', '.png')

all_paths = set()

for category in os.listdir("train_data"):
    category_path = os.path.join("train_data", category)

    if not os.path.isdir(category_path):
        continue

    for filename in os.listdir(category_path):

        if filename.lower().endswith(exts):
            rel_path = f"{category}/{filename}"
            all_paths.add(rel_path)

# -----------------------------
# 3️⃣ 누락 이미지 찾기
# -----------------------------
missing = sorted(all_paths - csv_paths)

print(f"\n📂 전체 이미지: {len(all_paths)}")
print(f"📄 CSV 데이터: {len(csv_paths)}")
print(f"❗ 누락 이미지: {len(missing)}")

print("\n" + "="*60)
print("❗ CSV에 없는 이미지 목록")
print("="*60)

for m in missing:
    print(m)

# -----------------------------
# 4️⃣ 누락 확장자 통계
# -----------------------------
ext_counter = Counter([os.path.splitext(f)[1].lower() for f in missing])

print("\n📊 누락 이미지 확장자 분포")
for k,v in ext_counter.items():
    print(f"{k} : {v}")

# -----------------------------
# 5️⃣ 누락 카테고리 통계
# -----------------------------
cat_counter = Counter([m.split("/")[0] for m in missing])

print("\n📊 누락 이미지 카테고리 분포")
for k,v in cat_counter.items():
    print(f"{k} : {v}")

print("\n✅ 누락 검사 완료")

# ======================================================
# ⭐ 전체 train_data 확장자 통계 (숨겨진 파일 찾기)
# ======================================================
print("\n" + "="*60)
print("📊 train_data 전체 확장자 통계")
print("="*60)

all_ext_counter = Counter()

for r, d, files in os.walk("train_data"):
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        all_ext_counter[ext] += 1

for k, v in all_ext_counter.items():
    print(f"{k} : {v}")

print("\n✅ 전체 검사 종료")
