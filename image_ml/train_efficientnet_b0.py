import os
import json
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm
from tqdm import tqdm


@dataclass
class CFG:
    train_dir: str = "data/train" #학습 데이터 경로
    val_dir: str = "data/val" #검증 데이터 경로
    out_dir: str = "checkpoints" #모델 저장 폴더
    model_name: str = "efficientnet_b0" 

    img_size: int = 224 #입력 이미지 크기
    batch_size: int = 32 #한 번에 학습시키는 이미지 수
    num_workers: int = 4 #병렬로 이미지 읽는 프로레스 수

    lr: float = 3e-4 #학습률(가중치 얼마나 크게 업데이트할지)
    weight_decay: float = 1e-4 #과적합 방지용 규제
    epochs_head: int = 5          # head만 학습(몇 번 돌릴지)
    epochs_finetune: int = 0      # 전체(or 일부) 미세조정 (0이면 생략)
    finetune_lr: float = 1e-5

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(cfg: CFG) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    train_tf = transforms.Compose([ #학습 전처리
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(cfg.img_size * 1.15)),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    train_ds = datasets.ImageFolder(cfg.train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(cfg.val_dir, transform=val_tf)

    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    return train_loader, val_loader, idx_to_class


def build_model(cfg: CFG, num_classes: int) -> nn.Module:
    model = timm.create_model(cfg.model_name, pretrained=True, num_classes=num_classes)
    return model


def freeze_backbone(model: nn.Module):
    # timm 모델은 보통 classifier만 학습시키면 됨
    for name, p in model.named_parameters():
        p.requires_grad = False
    # classifier만 다시 True
    for name, p in model.named_parameters():
        if "classifier" in name or "fc" in name or "head" in name:
            p.requires_grad = True


def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = crit(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def train_one_epoch(model: nn.Module, loader: DataLoader, opt, device: str) -> Tuple[float, float]:
    model.train()
    crit = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    cfg = CFG()
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    train_loader, val_loader, idx_to_class = build_loaders(cfg)
    num_classes = len(idx_to_class)

    model = build_model(cfg, num_classes=num_classes).to(cfg.device)

    # 1) head만 학습
    freeze_backbone(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_acc = -1.0
    best_path = os.path.join(cfg.out_dir, "best_efficientnet_b0.pt")

    for epoch in range(1, cfg.epochs_head + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, cfg.device)
        va_loss, va_acc = evaluate(model, val_loader, cfg.device)

        print(f"[HEAD] epoch {epoch}/{cfg.epochs_head} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "model_name": cfg.model_name,
                "state_dict": model.state_dict(),
                "idx_to_class": idx_to_class,
                "img_size": cfg.img_size,
            }, best_path)
            print(f"  -> saved best: {best_path} (val acc {best_acc:.4f})")

    # 2) (선택) 전체 미세조정
    if cfg.epochs_finetune > 0:
        print("\n[FINETUNE] unfreezing all layers...")
        unfreeze_all(model)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.finetune_lr, weight_decay=cfg.weight_decay)

        for epoch in range(1, cfg.epochs_finetune + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, cfg.device)
            va_loss, va_acc = evaluate(model, val_loader, cfg.device)

            print(f"[FT] epoch {epoch}/{cfg.epochs_finetune} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f}")

            if va_acc > best_acc:
                best_acc = va_acc
                torch.save({
                    "model_name": cfg.model_name,
                    "state_dict": model.state_dict(),
                    "idx_to_class": idx_to_class,
                    "img_size": cfg.img_size,
                }, best_path)
                print(f"  -> saved best: {best_path} (val acc {best_acc:.4f})")

    # 클래스 매핑도 별도 저장(편의)
    mapping_path = os.path.join(cfg.out_dir, "idx_to_class.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx_to_class.items()}, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Best val acc = {best_acc:.4f}")
    print(f"Saved mapping: {mapping_path}")


if __name__ == "__main__":
    main()
