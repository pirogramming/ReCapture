# predict_text_gate.py
# 예측 + 신뢰도 판단(게이트)
# 라벨 4종(finance, study_note, info, others) 기준

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CKPT_DIR = "checkpoints_text"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 텍스트 게이트 기준(초기값)
CONF_TH = 0.60
MARGIN_TH = 0.10

# 라벨 고정(팀플 사고 방지)
EXPECTED_LABELS = ["finance", "study_note", "info", "others"]

# (중요) 모델/토크나이저는 한 번만 로드
_tokenizer = None
_model = None


def load_text_model():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)
    if _model is None:
        _model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR).to(DEVICE).eval()
    return _tokenizer, _model


def need_fallback_text(result, conf_th=CONF_TH, margin_th=MARGIN_TH):
    return (result["top1_prob"] < conf_th) or (result["margin"] < margin_th)


def _idx_to_label(id2label, idx: int) -> str:
    # id2label이 dict(str키)일 수도, list일 수도 있어서 안전 처리
    if isinstance(id2label, dict):
        return id2label.get(str(idx), id2label.get(idx, str(idx)))
    return id2label[idx]


def _validate_labels(model):
    # 모델이 가진 라벨이 expected와 동일한지 체크(순서/집합 둘 다)
    id2label = model.config.id2label
    model_labels = [_idx_to_label(id2label, i) for i in range(model.config.num_labels)]

    if set(model_labels) != set(EXPECTED_LABELS):
        raise ValueError(
            "모델 라벨 구성이 예상과 다릅니다.\n"
            f"expected: {EXPECTED_LABELS}\n"
            f"model_labels: {model_labels}\n\n"
            "원인: (1) 학습 데이터 label 오타 (2) 라벨 고정 안 하고 학습함 (3) 다른 ckpt 폴더를 로드함"
        )

    # 순서까지도 기대하는 순서로 강제하고 싶으면 아래 주석 해제
    # if model_labels != EXPECTED_LABELS:
    #     print("⚠️ WARNING: 라벨 순서가 EXPECTED_LABELS와 다릅니다.")
    #     print("expected order:", EXPECTED_LABELS)
    #     print("model order:", model_labels)


@torch.no_grad()
def predict_text(text: str, topk: int = 4):
    tokenizer, model = load_text_model()
    _validate_labels(model)

    enc = tokenizer(
        (text or "").strip(),
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    logits = model(**enc).logits  # (1, num_labels)
    probs = F.softmax(logits, dim=1).squeeze(0)  # (num_labels,)

    # top-k 랭킹
    k = min(topk, probs.numel())
    top = torch.topk(probs, k=k)

    id2label = model.config.id2label
    topk_list = []
    for idx, p in zip(top.indices.tolist(), top.values.tolist()):
        topk_list.append({"label": _idx_to_label(id2label, int(idx)), "prob": float(p)})

    # top1 / top2 / margin
    top1 = topk_list[0]
    top2 = topk_list[1] if len(topk_list) > 1 else {"label": top1["label"], "prob": 0.0}
    margin = float(top1["prob"] - top2["prob"])

    return {
        "top1_label": top1["label"],
        "top1_prob": float(top1["prob"]),
        "top2_label": top2["label"],
        "top2_prob": float(top2["prob"]),
        "margin": margin,
        "topk": topk_list,
    }


def main():
    # 테스트용 텍스트(여기 바꿔가며 확인)
    text = """
    결제 승인
    금액 12,000원
    카드
    """

    out = predict_text(text, topk=4)
    print("TEXT ML RESULT:", out)

    if need_fallback_text(out):
        print("→ LOW CONFIDENCE (text): consider manual review or alternative rule/LLM")
    else:
        print("→ HIGH CONFIDENCE (text): accept text model result")


if __name__ == "__main__":
    main()
