# build_text_json_from_clova.py
# Clova OCR JSON → 텍스트 분류 학습용 jsonl(train/val) 생성기
# 라벨: finance / study_note / info / others

import os
import glob
import json

# OCR 결과 원본 폴더 (train/val 아래에 라벨별 폴더)
SRC_ROOT = "data_clova"
# 텍스트 ML 학습용 jsonl 출력 폴더
OUT_ROOT = "data_text"
os.makedirs(OUT_ROOT, exist_ok=True)

# 라벨 4종 고정 (팀플 사고 방지)
ALLOWED_LABELS = {"finance", "study_note", "info", "others"}

# OCR 텍스트가 너무 빈약할 때 스킵 기준 (공백 제거 후 글자수)
MIN_CHARS = 10


def clova_json_to_text(obj: dict) -> str:
    """
    Clova OCR 응답(JSON)에서 inferText를 모아 하나의 텍스트로 합친다.
    일반 구조: obj["images"][0]["fields"][i]["inferText"]
    """
    texts = []
    images = obj.get("images", [])
    if not images:
        return ""

    fields = images[0].get("fields", [])
    for f in fields:
        t = (f.get("inferText") or "").strip()
        if t:
            texts.append(t)

    # 줄바꿈으로 이어붙여 원문 느낌 유지
    return "\n".join(texts).strip()


def build_split(split: str):
    """
    split: "train" or "val"
    입력:  data_clova/{split}/{label}/*.json
    출력:  data_text/{split}.jsonl
    """
    out_path = os.path.join(OUT_ROOT, f"{split}.jsonl")
    base = os.path.join(SRC_ROOT, split)

    if not os.path.isdir(base):
        raise FileNotFoundError(f"폴더 없음: {base}")

    rows = 0
    skipped_short = 0
    skipped_bad_label_folder = 0
    skipped_bad_json = 0

    with open(out_path, "w", encoding="utf-8") as w:
        for label in sorted(os.listdir(base)):
            label_dir = os.path.join(base, label)
            if not os.path.isdir(label_dir):
                continue

            # 라벨 폴더명 검증 (오타/옛 라벨 방지)
            if label not in ALLOWED_LABELS:
                print(f"[WARN] skip unknown label folder: {label_dir}")
                skipped_bad_label_folder += 1
                continue

            for fp in glob.glob(os.path.join(label_dir, "*.json")):
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                except Exception:
                    skipped_bad_json += 1
                    continue

                text = clova_json_to_text(obj)

                # OCR이 너무 빈약한 경우 스킵(학습 품질 방어)
                text_compact = "".join((text or "").split())
                if len(text_compact) < MIN_CHARS:
                    skipped_short += 1
                    continue

                # src를 같이 저장하면 나중에 디버깅/역추적이 쉬움
                w.write(
                    json.dumps(
                        {"text": text, "label": label, "src": fp},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rows += 1

    print(f"[{split}] saved -> {out_path} ({rows} rows)")
    print(f"  - skipped_short(<{MIN_CHARS} chars): {skipped_short}")
    print(f"  - skipped_bad_label_folders: {skipped_bad_label_folder}")
    print(f"  - skipped_bad_json: {skipped_bad_json}")


def main():
    build_split("train")
    build_split("val")


if __name__ == "__main__":
    main()
