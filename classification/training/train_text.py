import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import pandas as pd

# 텍스트 모델 학습 파일(train_text.csv 읽어서 koELECTRA 텍스트 분류 모델 학습)
# 결과 저장: models/text_model_v1 (+label_map.json)
# ✅ 출력/흐름은 기존과 동일, 카테고리만 4개로 고정

# -----------------------------------------------------------
# 1. 카테고리 정의 ✅ 4개로 수정
# -----------------------------------------------------------
VALID_CATEGORIES = ["finance", "study_note", "info", "others"]
LABEL_TO_IDX = {cat: idx for idx, cat in enumerate(VALID_CATEGORIES)}


class TextDataset(Dataset):
    """CSV 파일에서 텍스트를 읽어오는 데이터셋 (OCR 과정 없음!)"""

    def __init__(self, csv_file, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"📂 데이터셋 로딩 중: {csv_file}")
        self.df = pd.read_csv(csv_file)

        # 기본 컬럼 체크
        if "category" not in self.df.columns or "text" not in self.df.columns:
            raise ValueError("❌ train_text.csv에 'category', 'text' 컬럼이 필요합니다.")

        # category 정리
        self.df["category"] = self.df["category"].astype(str)

        # ✅ 허용 카테고리 검증 (4개만)
        unknown = sorted(set(self.df["category"].unique()) - set(VALID_CATEGORIES))
        if unknown:
            raise ValueError(
                f"❌ train_text.csv에 허용되지 않은 category가 있어요: {unknown}\n"
                f"✅ 허용 카테고리(4개): {VALID_CATEGORIES}"
            )

        # 라벨 인코딩 정보
        self.categories = VALID_CATEGORIES
        self.label_to_idx = LABEL_TO_IDX

        print(f"✅ 카테고리 맵핑(4개): {self.label_to_idx}")
        print(f"✅ 총 데이터 개수: {len(self.df)}개")
        print(f"✅ 라벨 분포:\n{self.df['category'].value_counts()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row.get("text", ""))
        category = row["category"]
        label = self.label_to_idx[category]

        # 너무 짧은 텍스트 처리(기존 스타일 유지)
        if len(text.strip()) < 2:
            text = "내용 없음"

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def train_model():
    print("=" * 70)
    print("🤖 텍스트 모델 학습 시작 (KoELECTRA-Base)")
    print("=" * 70)

    model_name = "monologg/koelectra-base-v3-discriminator"
    csv_path = "train_text.csv"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    full_dataset = TextDataset(csv_path, tokenizer)
    num_labels = len(full_dataset.categories)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    train_loader = DataLoader(full_dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✅ 디바이스: {device}")
    print(f"✅ num_labels: {num_labels}")

    optimizer = AdamW(model.parameters(), lr=3e-5)

    num_epochs = 10
    print(f"\n🔥 학습 시작 (Epochs: {num_epochs})")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100.0 * correct / max(total, 1)
        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    save_dir = "models/text_model_v1"
    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # ✅ label_map.json 저장(4개 고정)
    with open(os.path.join(save_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(full_dataset.label_to_idx, f, ensure_ascii=False, indent=4)

    print(f"💾 모델 저장 완료: {save_dir}")
    print("✅ label_map.json 저장 완료")


if __name__ == "__main__":
    train_model()
