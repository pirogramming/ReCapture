# train_image.py
# ✅ HEIC/HEIF 포함 학습 + (중요) 깨진/인식불가 이미지 자동 제외 버전
# - pillow-heif 등록
# - Image.open(path)로 열어서 HEIC 인식 안정화
# - 학습 시작 전, 열 수 없는 파일은 dataset에서 제거(학습 중단 방지)

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torchvision import transforms
from torchvision.datasets import ImageFolder

import timm
from PIL import Image
from PIL import UnidentifiedImageError

# ✅ HEIC 지원
from pillow_heif import register_heif_opener
register_heif_opener()


# ======================
# 설정값(여기만 바꿔도 됨)
# ======================
DATA_DIR = "train_data"          # 학습 데이터 폴더 (카테고리별 하위 폴더)
SAVE_PATH = "models/efficientnet_v1.pth"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
IMG_SIZE = 224
VAL_RATIO = 0.2
NUM_WORKERS = 0                 # Windows면 0이 안전

# ✅ 최종 카테고리 4개(폴더명 = label)
VALID_CATEGORIES = ["finance", "study_note", "info", "others"]


# ======================
# HEIC 포함 ImageFolder
# ======================
HEIC_EXTS = {".heic", ".heif"}

def pil_loader(path: str):
    """
    ✅ 중요: HEIC는 file object(Image.open(f))로 열면 실패하는 경우가 있어서
    무조건 path 문자열로 Image.open(path) 사용.
    """
    img = Image.open(path)      # ← 핵심 변경
    return img.convert("RGB")

def is_valid_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"} | HEIC_EXTS

class ImageFolderWithHEIC(ImageFolder):
    def __init__(self, root, transform=None, strict_filter=True):
        super().__init__(
            root=root,
            transform=transform,
            loader=pil_loader,
            is_valid_file=is_valid_file,
        )
        # ✅ 학습 전에 "열 수 없는 파일" 자동 제거 (깨진 HEIC/이상한 파일 때문에 학습 멈추는 것 방지)
        if strict_filter:
            self._filter_bad_samples()

    def _filter_bad_samples(self):
        kept = []
        removed = []
        for p, y in self.samples:
            try:
                # 실제로 열어보기(가벼운 검증)
                img = Image.open(p)
                img.verify()  # 파일 무결성 체크(빠름)
                kept.append((p, y))
            except (UnidentifiedImageError, OSError, ValueError) as e:
                removed.append((p, str(e)))

        if removed:
            print(f"⚠️ 열 수 없는 이미지 {len(removed)}개 발견 → 학습에서 제외합니다.")
            # 너무 많이 찍히면 로그 폭발하니까 상위 10개만
            for p, msg in removed[:10]:
                print(f" - 제외: {p} ({msg})")
            if len(removed) > 10:
                print(f" - ... 외 {len(removed)-10}개 더")

        self.samples = kept
        self.imgs = kept  # torchvision ImageFolder 호환


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"❌ DATA_DIR 폴더가 없습니다: {DATA_DIR}\n"
            f"예) {DATA_DIR}/finance, {DATA_DIR}/study_note, {DATA_DIR}/info, {DATA_DIR}/others"
        )

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ✅ HEIC 포함 dataset + 깨진 파일 자동 제외
    full_dataset = ImageFolderWithHEIC(DATA_DIR, transform=transform, strict_filter=True)
    num_classes = len(full_dataset.classes)

    # ✅ (추가) 폴더/카테고리 4개 검증 (출력 흐름은 그대로, 이상할 때만 에러/경고)
    unknown = [c for c in full_dataset.classes if c not in VALID_CATEGORIES]
    missing = [c for c in VALID_CATEGORIES if c not in full_dataset.classes]

    if unknown:
        raise ValueError(
            f"❌ train_data 안에 알 수 없는 폴더가 있어요: {unknown}\n"
            f"✅ 허용 폴더(4개): {VALID_CATEGORIES}"
        )

    if missing:
        # 실수 방지용 경고(기존 make_dataset 방식과 동일한 톤)
        print(f"⚠️ train_data 안에 없는(누락된) 카테고리 폴더: {missing}")
        print("   (비어있어도 괜찮으면 빈 폴더라도 만들어두는 걸 추천)")

    if sorted(full_dataset.classes) != sorted(VALID_CATEGORIES):
        raise ValueError(
            f"❌ 카테고리 폴더가 정확히 4개여야 합니다.\n"
            f"   기대: {VALID_CATEGORIES}\n"
            f"   현재: {full_dataset.classes}"
        )

    if num_classes < 2:
        raise ValueError(
            f"❌ 카테고리 폴더가 너무 적습니다. (classes={full_dataset.classes})\n"
            f"{DATA_DIR} 아래에 2개 이상 카테고리 폴더가 있어야 학습이 가능합니다."
        )

    print(f"📦 Classes ({num_classes}): {full_dataset.classes}")
    print(f"🖼️ Total images (after filter): {len(full_dataset)}")

    val_size = max(1, int(len(full_dataset) * VAL_RATIO))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
    )

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(SAVE_PATH) or ".", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ---- train ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        # ---- val ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        dt = time.time() - t0
        print(
            f"[Epoch {epoch:02d}/{EPOCHS}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% | "
            f"{dt:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"💾 Best model saved -> {SAVE_PATH} (val_acc={best_val_acc*100:.2f}%)")

    print(f"\n✅ Training done. Best val_acc={best_val_acc*100:.2f}%")
    print("ℹ️ 클래스 순서(full_dataset.classes)는 추론 시에도 동일하게 맞춰야 합니다.")


if __name__ == "__main__":
    main()
