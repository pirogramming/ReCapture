import os
import glob
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm

import os
import glob
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm


# ======================
# 설정값(여기만 바꿔도 됨)
# ======================
TEST_DIR = "test_images"  # 테스트 이미지 폴더
CKPT_PATH = "effv2b0_recapture_best.pt"  # 학습된 체크포인트
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# OCR 게이트 기준(초기값)
CONF_TH = 0.75
MARGIN_TH = 0.15


def load_model(ckpt_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"체크포인트 파일이 없습니다: {ckpt_path}\n"
            f"- 먼저 학습을 돌려서 pt 파일을 만들거나, 경로를 맞춰주세요."
        )

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    model_name = ckpt.get("model_name", "tf_efficientnetv2_b0")

    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_to_idx))
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()

    return model, idx_to_class


def build_transform():
    # 학습 때 쓴 val_tf와 동일하게 맞추는 게 좋음
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]


@torch.no_grad()
def predict_one(model, tfm, idx_to_class: Dict[int, str], image_path: str) -> Dict:
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]  # shape: [C]

    topk = torch.topk(probs, k=min(3, probs.numel()))
    top_indices = topk.indices.tolist()
    top_scores = topk.values.tolist()

    top1_idx = int(top_indices[0])
    top1_conf = float(top_scores[0])
    top2_conf = float(top_scores[1]) if len(top_scores) > 1 else 0.0
    margin = top1_conf - top2_conf

    label = idx_to_class[top1_idx]

    # OCR 게이트: 확신 낮거나, 1/2등 차이 작으면 OCR
    ocr_on = (top1_conf < CONF_TH) or (margin < MARGIN_TH)

    # 확률 전체 dict (원하면 로그/디버깅에 사용)
    prob_dict = {idx_to_class[i]: float(probs[i]) for i in range(probs.numel())}

    # top3 보기 좋게
    top3 = [(idx_to_class[int(i)], float(s)) for i, s in zip(top_indices, top_scores)]

    return {
        "path": image_path,
        "label": label,
        "conf": top1_conf,
        "margin": margin,
        "ocr_on": ocr_on,
        "top3": top3,
        "probs": prob_dict,
    }


def iter_images(test_dir: str) -> List[str]:
    paths = []
    for p in glob.glob(os.path.join(test_dir, "**", "*"), recursive=True):
        if os.path.isfile(p) and is_image_file(p):
            paths.append(p)
    paths.sort()
    return paths


def main():
    model, idx_to_class = load_model(CKPT_PATH)
    tfm = build_transform()

    paths = iter_images(TEST_DIR)
    if not paths:
        print(f"[WARN] '{TEST_DIR}' 폴더에서 이미지를 찾지 못했습니다.")
        return

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Classes: {list(sorted(set(idx_to_class.values())))}")
    print(f"Gate: conf<{CONF_TH} OR margin<{MARGIN_TH} => OCR_ON\n")

    for p in paths:
        out = predict_one(model, tfm, idx_to_class, p)

        fname = os.path.relpath(out["path"], TEST_DIR)
        label = out["label"]
        conf = out["conf"]
        margin = out["margin"]
        ocr_on = out["ocr_on"]
        top3 = out["top3"]

        print(f"- {fname}")
        print(f"  pred: {label} | conf={conf:.4f} | margin={margin:.4f} | OCR_ON={ocr_on}")
        print(f"  top3: " + ", ".join([f"{k}:{v:.4f}" for k, v in top3]))
        print("")

if __name__ == "__main__":
    main()
