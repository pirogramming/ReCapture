def classify_one(photo_path: str) -> dict:
    # 1) OCR
    ocr_text = extract_text(photo_path)  # PaddleOCR

    # 2) text model
    p_text = predict_text_probs(ocr_text)  # DistilBERT softmax

    # 3) image model
    p_img = predict_img_probs(photo_path)  # MobileNetV3 softmax

    # 4) ensemble
    alpha = dynamic_alpha(ocr_text)
    p = alpha * p_text + (1 - alpha) * p_img

    label_id = int(p.argmax())
    confidence = float(p[label_id])

    return {
        "label": LABELS[label_id],
        "confidence": confidence,
        "alpha": alpha,
        "ocr_text": ocr_text[:5000],  # 너무 길면 잘라 저장
    }
