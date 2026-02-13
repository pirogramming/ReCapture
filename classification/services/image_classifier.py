# services/image_classifier.py
import os
from pathlib import Path
from django.conf import settings
from typing import List, Tuple

import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms

# ✅ HEIC 지원
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass


class ImageClassifier:
    """
    ✅ EfficientNet 이미지 분류기 (최종 4카테고리)
    - predict(): (top1_cat, top1_conf)
    - predict_proba(): (top1_cat, top1_conf, top2_cat, top2_conf, margin)
    """

    # ✅ 4카테고리(폴더명/라벨명과 동일)
    # ⚠️ 순서는 "학습 때 ImageFolder가 만든 classes 순서"와 같아야 함.
    #    (보통 알파벳순: finance, info, others, study_note)
    CATEGORIES: List[str] = [
        "finance",
        "info",
        "others",
        "study_note",
    ]

    CATEGORY_KR = {
        "finance": "결제/예약",
        "study_note": "학습/노트",
        "info": "정보",
        "others": "기타",
    }

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        model_path: str = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        # ✅ (출력 형태 유지)
        print("🔧 이미지 분류 모델 로딩 중...")


        # ✅ model_path가 None이면 프로젝트 기준 절대경로로 기본값 설정
        if model_path is None:
            model_path = str(Path(settings.BASE_DIR) / "classification" / "models" / "efficientnet_v1.pth")

        if not os.path.exists(model_path):
            alt = str(Path(settings.BASE_DIR) / "saved_models" / "efficientnet_v1.pth")
            if os.path.exists(alt):
                model_path = alt
            else:
                raise FileNotFoundError(
                    f"❌ 모델 파일이 없습니다: {model_path}\n"
                    f"cwd={os.getcwd()}"
                )

        print(f"   📂 모델 파일: {model_path}")

        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=len(self.CATEGORIES),
        ).to(self.device)
        self.model.eval()

        ckpt = torch.load(model_path, map_location=self.device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if isinstance(ckpt, dict):
            ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

        self.model.load_state_dict(ckpt, strict=True)

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.categories = self.CATEGORIES[:]
        self.category_kr = self.CATEGORY_KR.copy()

        print(f"✅ 모델 준비 완료! (카테고리: {self.categories})\n")

    @torch.no_grad()
    def predict(self, image_path: str) -> Tuple[str, float]:
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

        top_idx = int(torch.argmax(probs).item())
        conf = float(probs[top_idx].item())
        return self.categories[top_idx], conf

    @torch.no_grad()
    def predict_proba(self, image_path: str):
        """
        Returns:
            top1_cat, top1_conf, top2_cat, top2_conf, margin
        """
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

        top2 = torch.topk(probs, k=2)
        idx1 = int(top2.indices[0].item())
        idx2 = int(top2.indices[1].item())
        p1 = float(top2.values[0].item())
        p2 = float(top2.values[1].item())

        c1 = self.categories[idx1]
        c2 = self.categories[idx2]
        margin = p1 - p2
        return c1, p1, c2, p2, margin

    def predict_with_korean(self, image_path: str):
        cat, conf = self.predict(image_path)
        return cat, self.category_kr.get(cat, cat), float(conf)
