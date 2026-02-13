# services/text_classifier.py
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from django.conf import settings

class TextClassifier:
    """
    🧠 텍스트 분류기 (✅ 4카테고리 기준)

    - models/text_model_v1/label_map.json을 우선 신뢰
      * 4개면 4개로 동작
    - label_map.json이 없으면 DEFAULT_4로 임시 설정
    """

    # ✅ label_map.json이 없을 때의 기본값(4개)
    DEFAULT_4 = [
        "finance",
        "study_note",
        "info",
        "others",
    ]

    CATEGORY_KR_4 = {
        "finance": "결제/예약",
        "study_note": "학습/노트",
        "info": "정보",
        "others": "기타",
    }

    def __init__(self):
        # ✅ (출력 형태 유지)
        print("🔧 텍스트 분류 모델 로딩 중...")

        self.model_path = self.model_path = str(Path(settings.BASE_DIR) / "classification" / "models" / "text_model_v1")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"❌ 모델 폴더가 없습니다: {self.model_path}\n"
                f"먼저 training/train_text.py를 실행해서 모델을 학습시켜주세요!"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ✅ 라벨맵 로드(있으면 그걸 그대로 사용)
        label_map_path = os.path.join(self.model_path, "label_map.json")
        if os.path.exists(label_map_path):
            with open(label_map_path, "r", encoding="utf-8") as f:
                label_map = json.load(f)  # 보통 {"label": id} 형태

            # {"finance":0,...} -> id_to_label
            self.id_to_label = {int(v): k for k, v in label_map.items()}
            print(f"✅ 라벨 맵핑 로드 완료: {len(self.id_to_label)}개 카테고리")
        else:
            # ✅ 없으면 4개 기본값으로 가정
            print("⚠️ label_map.json이 없습니다 → DEFAULT_4로 임시 설정합니다.")
            self.id_to_label = {i: cat for i, cat in enumerate(self.DEFAULT_4)}

        # 모델/토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        # ✅ (출력 형태 유지)
        print(f"✅ 모델 로드 성공 (Device: {self.device})")

    def predict(self, text: str):
        """텍스트 -> (category, confidence)"""
        if not text or len(text.strip()) < 2:
            return "others", 0.0

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        pred_idx = int(pred_idx.item())
        confidence = float(confidence.item())

        # 안전 처리: id_to_label에 없으면 others
        category = self.id_to_label.get(pred_idx, "others")
        return category, confidence

    def predict_with_korean(self, text: str):
        cat, conf = self.predict(text)
        kr = self.CATEGORY_KR_4.get(cat, cat)
        return cat, kr, conf
