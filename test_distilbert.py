# test_distilbert.py
from transformers import pipeline

LABELS = ["finance", "study_note", "shopping", "schedule", "document", "others"]

def judge_confidence(score1: float, score2: float) -> str:
    """
    아주 단순한 룰:
    - score1이 높고(score>=0.65), 1~2등 차이도 충분하면 확신 높음
    - 그 외는 애매/낮음
    """
    margin = score1 - score2
    if score1 >= 0.70 and margin >= 0.15:
        return "HIGH"
    if score1 >= 0.55 and margin >= 0.08:
        return "MEDIUM"
    return "LOW"

def main():
    clf = pipeline(
        task="zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli",
        device=-1,  # CPU
    )

    text = """
    결제 승인
    금액 12,000원
    카드
    """

    out = clf(text, LABELS, multi_label=False)

    top1_label = out["labels"][0]
    top1_score = float(out["scores"][0])
    top2_score = float(out["scores"][1]) if len(out["scores"]) > 1 else 0.0
    margin = top1_score - top2_score
    level = judge_confidence(top1_score, top2_score)

    print("=== INPUT TEXT ===")
    print(text.strip())

    print("\n=== PREDICTION ===")
    print(f"label      : {top1_label}")
    print(f"confidence : {top1_score:.4f} (0~1)")
    print(f"margin     : {margin:.4f} (top1-top2)")
    print(f"level      : {level}")

    print("\n=== ALL SCORES ===")
    for label, score in zip(out["labels"], out["scores"]):
        print(f"{label:>10s} : {float(score):.4f}")

if __name__ == "__main__":
    main()
