#예측 전용(예측만 함)
#텍스트 분류 모델을 만드는 코드
import os
import json
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score


# ======================
# 설정값(여기만 바꾸면 됨)
# ======================
@dataclass
class CFG:
    train_path: str = "data_text/train.jsonl"
    val_path: str = "data_text/val.jsonl"
    out_dir: str = "checkpoints_text"
    model_name: str = "monologg/koelectra-base-v3-discriminator"  # 추천 KoELECTRA
    max_length: int = 256
    batch_size: int = 16
    lr: float = 2e-5
    epochs: int = 3
    weight_decay: float = 0.01
    seed: int = 42


# ======================
# 라벨 고정(중요!)
# ======================
LABELS = ["finance", "study_note", "info", "others"]
label2id = {lb: i for i, lb in enumerate(LABELS)}
id2label = {i: lb for lb, i in label2id.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


def main():
    cfg = CFG()
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 1) 데이터 로드 (jsonl)
    ds = load_dataset(
        "json",
        data_files={"train": cfg.train_path, "validation": cfg.val_path},
    )

    # (선택) 데이터에 라벨 오타가 있으면 여기서 바로 터지게 해서 사고 방지
    invalid = sorted(list(set(ds["train"]["label"]) - set(LABELS)))
    if invalid:
        raise ValueError(
            f"train.jsonl에 허용되지 않은 label이 있습니다: {invalid}\n"
            f"허용 label: {LABELS}"
        )
    invalid_val = sorted(list(set(ds["validation"]["label"]) - set(LABELS)))
    if invalid_val:
        raise ValueError(
            f"val.jsonl에 허용되지 않은 label이 있습니다: {invalid_val}\n"
            f"허용 label: {LABELS}"
        )

    # 2) 토크나이저/모델
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    # 3) 전처리(토크나이즈)
    def preprocess(ex):
        text = (ex["text"] or "").strip()
        enc = tokenizer(
            text,
            truncation=True,
            max_length=cfg.max_length,
            padding=False,
        )
        enc["labels"] = label2id[ex["label"]]
        return enc

    ds_tok = ds.map(preprocess, remove_columns=ds["train"].column_names)

    # 4) Trainer 설정
    args = TrainingArguments(
        output_dir=cfg.out_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=cfg.seed,
        fp16=torch.cuda.is_available(),  # GPU면 속도↑
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # 5) best 모델 저장 + 라벨 매핑 저장
    trainer.save_model(cfg.out_dir)
    tokenizer.save_pretrained(cfg.out_dir)

    with open(os.path.join(cfg.out_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print("Saved to:", cfg.out_dir)
    print("Labels:", LABELS)
    print("label2id:", label2id)


if __name__ == "__main__":
    main()
