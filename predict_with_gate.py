
#이미지 ML ->텍스트로 넘길지
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm


def load_model(ckpt_path: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)

    num_classes = len(ckpt["idx_to_class"])
    model = timm.create_model(
        ckpt["model_name"],
        pretrained=False,
        num_classes=num_classes,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    idx_to_class = ckpt["idx_to_class"]
    img_size = ckpt.get("img_size", 224)

    tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    return model, idx_to_class, tf, device


@torch.no_grad()
def predict_one(model, idx_to_class, tf, device, image_path: str):
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0)

    top2 = torch.topk(probs, k=2)
    top1_idx = int(top2.indices[0])
    top2_idx = int(top2.indices[1])

    top1_prob = float(top2.values[0])
    top2_prob = float(top2.values[1])

    return {
        "top1_label": idx_to_class[top1_idx],
        "top1_prob": top1_prob,
        "top2_label": idx_to_class[top2_idx],
        "top2_prob": top2_prob,
        "margin": top1_prob - top2_prob,
    }


def need_text_ml(result, prob_th=0.65, margin_th=0.15) -> bool:
    # 둘 중 하나라도 만족하면 "애매" → OCR + 텍스트 ML로 넘기기
    return (result["top1_prob"] < prob_th) or (result["margin"] < margin_th)


def main():
    ckpt_path = "checkpoints/best_efficientnet_b0.pt"  # 학습 후 생기는 파일
    image_path = "data/val/study_note/001.jpg"         # 테스트할 이미지로 바꿔줘

    model, idx_to_class, tf, device = load_model(ckpt_path)
    out = predict_one(model, idx_to_class, tf, device, image_path)

    print("IMAGE ML RESULT:", out)

    if need_text_ml(out, prob_th=0.65, margin_th=0.15):
        print("→ LOW CONFIDENCE: send to OCR → TEXT ML")
    else:
        print("→ HIGH CONFIDENCE: accept image model result")


if __name__ == "__main__":
    main()
