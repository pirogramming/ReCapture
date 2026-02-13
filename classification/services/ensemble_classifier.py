# services/ensemble_classifier.py
import os
import sys
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.google_ocr_service import GoogleOCRService
from services.image_classifier import ImageClassifier
from services.text_classifier import TextClassifier
from services.ocr_cache import OCRCache


class EnsembleClassifier:
    """
    ✅ 최종 튜닝 (Shopping vs Finance Boundary Fix)
    - '쇼핑몰 앱 화면(상품상세, 주문목록)'은 Info로 분류되도록 UI 키워드 강화
    - '순수 영수증/뱅킹'만 Finance로 남도록 조건 정교화
    """

    CATS = ["finance", "study_note", "info", "others"]

    CATEGORY_KR = {
        "finance": "결제/예약",
        "study_note": "학습/노트",
        "info": "정보",
        "others": "기타",
    }

    OCR_STRONG = {"finance", "info", "study_note"}

    # 텍스트 신뢰도 임계값
    TEXT_OVERRIDE_TH = {
        "finance": 0.88,    # 쇼핑몰 오분류 방지를 위해 높게 유지
        "info": 0.80,
        "study_note": 0.80,
        "others": 0.95,
    }

    def __init__(self):
        print("🚀 통합 분류 시스템 초기화 중 (Shopping/Finance Boundary Fix)...\n")
        try:
            self.ocr = GoogleOCRService()
        except Exception as e:
            print(f"⚠️ Google OCR 초기화 실패: {e}")
            self.ocr = None

        self.ocr_cache = OCRCache(cache_dir="ocr_cache")
        self.image_clf = ImageClassifier()

        try:
            self.text_clf = TextClassifier()
        except Exception as e:
            print(f"⚠️ TextClassifier 로드 실패: {e}")
            self.text_clf = None

        self.ocr_triggers = ["finance", "info", "study_note"]
        print("✅ 모든 모델 로드 완료!\n")

    def _get_ocr_text_cached(self, image_path: str, logs: list) -> str:
        cached = self.ocr_cache.load(image_path)
        if cached is not None:
            logs.append("⚡ OCR CACHE HIT")
            return cached.get("full_text", "") if isinstance(cached, dict) else str(cached)

        if self.ocr is None:
            logs.append("⚠️ OCR 서비스 미작동")
            return ""

        logs.append("🧠 OCR CACHE MISS")
        ocr_result = self.ocr.extract_text(image_path)
        if isinstance(ocr_result, dict):
            self.ocr_cache.save(image_path, ocr_result)
            return ocr_result.get("full_text", "")
        else:
            self.ocr_cache.save(image_path, {"full_text": str(ocr_result)})
            return str(ocr_result)

    def _text_hint_scores(self, text: str):
        t = (text or "")
        tl = t.lower()
        scores = {"finance": 0, "study_note": 0, "info": 0}

        # -----------------------
        # 1. Shopping UI (Info) - 가장 먼저 감지
        # -----------------------
        # 이 키워드들이 많으면 Finance일 확률을 낮춰야 함 (상품 페이지, 주문 목록 등)
        shopping_kws = [
            "장바구니", "상세정보", "리뷰", "구매하기", "옵션", "판매자", 
            "배송", "무료배송", "찜하기", "별점", "마이페이지", "cart",
            "q&a", "상품평", "즉시할인", "최소주문", "품절",
            "사이즈", "size", "컬러", "color", "색상", "용량", "구성품",
            "주문상세", "주문조회", "주문내역" # 이것들은 앱 화면(Info) 성격이 강함
        ]
        
        shopping_score = 0
        for kw in shopping_kws:
            if kw in tl:
                shopping_score += 1
                scores["info"] += 4

        # -----------------------
        # 2. Finance (결제/뱅킹/티켓)
        # -----------------------
        # [절대 키워드] 쇼핑몰에는 잘 없는 것들 (뱅킹, 티켓, 기프티콘)
        if re.search(r"통장|잔액|송금|이체|계좌|뱅킹|bank|예매|티켓|ticket|booking|탑승|체크인|선물하기|기프티콘|교환권", tl):
            scores["finance"] += 20 

        # [영수증] '영수증' 단어가 명시되면 Finance
        if "영수증" in tl or "receipt" in tl:
            scores["finance"] += 15

        # [일반 결제] 주문번호, 결제완료 등 (쇼핑몰에도 있을 수 있음)
        # -> 쇼핑몰 UI 점수가 높으면 이 점수를 무효화하거나 깎아야 함
        finance_kws = ["결제완료", "승인번호", "주문번호", "합계", "부가세", "vat", "total"]
        for kw in finance_kws:
            if kw in tl:
                scores["finance"] += 4
        
        if re.search(r"\d{1,3}(?:,\d{3})*원|\d+원", t):
            scores["finance"] += 1

        # [핵심 로직] 쇼핑몰 UI 징후가 뚜렷하면 Finance 점수 대폭 삭감
        # (단, '절대 키워드'가 +20점이므로 뱅킹/기프티콘은 살아남음)
        if shopping_score >= 1:
            scores["finance"] -= 8  # 삭감폭 강화 (-6 -> -8)
            scores["info"] += 5

        # 일반 정보/URL
        if re.search(r"https?://|www\.|\.com", tl):
            scores["info"] += 4
        
        # -----------------------
        # 3. Study Note
        # -----------------------
        for kw in ["정리", "정의", "증명", "theorem", "문제", "해설", "정답", "풀이", "chapter", "unit"]:
            if kw in tl: scores["study_note"] += 3

        return scores

    def classify(self, image_path: str):
        logs = []
        
        # Step 1. 비전 모델
        img_cat, img_conf, img2_cat, img2_conf, margin = self.image_clf.predict_proba(image_path)
        logs.append(f"👁️ 비전: {img_cat} ({img_conf*100:.1f}%), margin={margin:.3f}")

        final_cat, final_conf = img_cat, img_conf
        ocr_text = ""
        text_cat, text_conf = None, 0.0

        # Step 2. OCR 트리거
        is_trigger = img_cat in self.ocr_triggers
        is_uncertain = img_conf < 0.75
        is_ambiguous = margin < 0.20
        is_weak_others = (img_cat == "others" and img_conf < 0.90)

        if is_trigger or is_uncertain or is_ambiguous or is_weak_others:
            logs.append(f"🚨 OCR 진입 (U:{is_uncertain}, WeakOther:{is_weak_others})")
            
            if self.text_clf:
                ocr_text = self._get_ocr_text_cached(image_path, logs)
                cleaned = ocr_text.strip()
                tl_cleaned = cleaned.lower()

                # -----------------------
                # [Others 디테일] 텍스트가 거의 없는 경우
                # -----------------------
                if len(cleaned) < 3:
                    logs.append("❌ 텍스트 없음")
                    # 비전 모델이 80% 미만 확신이면 Others로 (배너, 풍경 등)
                    if img_conf < 0.80: 
                        final_cat = "others"
                        final_conf = 0.90
                        logs.append("🛡️ 텍스트 없음 + 비전 불확실 → Others 강제 할당")
                    else:
                        logs.append(f"ℹ️ 텍스트 없지만 비전({img_cat}) 신뢰 → 이미지 유지")
                else:
                    # Step 3. 텍스트 예측
                    text_cat, text_conf = self.text_clf.predict(cleaned)
                    logs.append(f"🧠 텍스트 모델: {text_cat} ({text_conf*100:.1f}%)")

                    # Step 3.5 힌트 보정
                    hints = self._text_hint_scores(cleaned)
                    best_hint = max(hints, key=hints.get)
                    hint_score = hints[best_hint]

                    if hint_score >= 4:
                        if hint_score >= 10: 
                            text_cat = best_hint
                            text_conf = 0.99
                            logs.append(f"🔥 강력한 힌트({best_hint}) 적용")
                        elif text_cat == best_hint:
                            text_conf = min(text_conf + 0.15, 0.99)
                        elif text_cat == "others" or text_conf < 0.8:
                            text_cat = best_hint
                            text_conf = 0.90
                            logs.append(f"✨ 힌트 기반 카테고리 수정: {best_hint}")

                    # ------------------------------------------------------------------
                    # [최종 판단 플래그] 
                    # ------------------------------------------------------------------
                    # 1. 절대적 Finance (뱅킹, 티켓, 기프티콘)
                    is_absolute_finance = bool(re.search(r"통장|잔액|송금|이체|계좌|예매|티켓|선물하기|기프티콘|교환권", tl_cleaned))
                    
                    # 2. 강력한 Shopping UI (상세정보, 리뷰, 장바구니 등)
                    is_shopping_ui = bool(re.search(r"상세정보|리뷰|장바구니|배송|옵션|판매자|주문상세|주문조회", tl_cleaned))

                    # Step 4. 최종 결정
                    th = self.TEXT_OVERRIDE_TH.get(text_cat, 0.85)
                    
                    # (A) 텍스트가 Others인 경우
                    if text_cat == "others":
                        if text_conf >= 0.95:
                            final_cat = "others"
                            final_conf = text_conf
                            logs.append("✅ 텍스트(others) 강력 확신 → 덮어쓰기")
                        elif img_cat in self.OCR_STRONG and img_conf > 0.85:
                            final_cat = img_cat
                            final_conf = img_conf
                            logs.append(f"🛡️ 텍스트 오분류(others) 방어: 비전({img_cat}) 유지")
                        elif text_conf >= th:
                            final_cat = text_cat
                            final_conf = text_conf
                            logs.append("✅ 텍스트(others) 적용")

                    # (B) 그 외 일반적인 경우
                    else:
                        # [Info vs Finance 충돌 해결]
                        if img_cat == "info" and text_cat == "finance":
                            # 1. 뱅킹/티켓/기프티콘 키워드가 있다? -> 무조건 Finance
                            if is_absolute_finance:
                                final_cat = "finance"
                                final_conf = text_conf
                                logs.append("🔓 절대적 금융 키워드 발견 → Finance 채택")
                            
                            # 2. 쇼핑몰 UI 키워드가 있다? -> 무조건 Info (Finance 방어)
                            elif is_shopping_ui:
                                final_cat = "info"
                                final_conf = 0.95 # 강제 상향
                                logs.append("🛡️ 쇼핑몰 UI 키워드 발견 → Info 채택")

                            # 3. 키워드는 없지만 비전(Info)이 확실하다 -> Info 유지
                            elif img_conf > 0.85:
                                logs.append("🛡️ 비전(info-쇼핑몰)이 확실하여 텍스트(finance) 무시")
                                final_cat = img_cat
                                final_conf = img_conf
                            
                            # 4. 정말 애매하면 Finance (안전빵)
                            else:
                                final_cat = "finance"
                                final_conf = text_conf
                                logs.append("⚖️ 애매함 → Finance 우선")

                        # [기프티콘 방어]
                        elif img_cat == "finance" and text_cat == "info":
                             if is_absolute_finance:
                                 logs.append("🛡️ 기프티콘/티켓 확실 → 텍스트(info) 무시하고 Finance 유지")
                                 final_cat = "finance"
                                 final_conf = max(img_conf, 0.95)
                             elif text_conf >= th:
                                 final_cat = text_cat
                                 final_conf = text_conf

                        elif text_conf >= th:
                            final_cat, final_conf = text_cat, text_conf
                            logs.append(f"✅ 텍스트({text_cat}) 확신 → 덮어쓰기")
                        elif text_cat == img_cat:
                            final_conf = min(img_conf + 0.1, 0.99)
                            logs.append("✅ 의견 일치")
                        else:
                            logs.append("ℹ️ 판단 보류 → 이미지 유지")

        return {
            "category": final_cat,
            "final_category_kr": self.CATEGORY_KR.get(final_cat, "알수없음"),
            "confidence": round(float(final_conf), 4),
            "image_result": {"category": img_cat, "confidence": float(img_conf)},
            "text_result": {"category": text_cat, "confidence": float(text_conf)},
            "ocr_text_preview": (ocr_text or "")[:100],
            "logs": logs,
        }



if __name__ == "__main__":
    clf = EnsembleClassifier()

    # ✅ 4카테고리 테스트 구조
    TEST_ROOT = "test_data"
    CATS = ["finance", "study_note", "info", "others"]
    EXTS = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif")

    PER_CLASS_LIMIT = None

    stats = {c: {"total": 0, "correct": 0} for c in CATS}
    confusion = {t: {p: 0 for p in CATS} for t in CATS}

    total = 0
    correct_total = 0

    img_correct_total = 0
    img_stats = {c: {"total": 0, "correct": 0} for c in CATS}
    
    text_correct_total = 0
    text_stats = {c: {"total": 0, "correct": 0} for c in CATS}

    if not os.path.isdir(TEST_ROOT):
        print(f"\n⚠️ {TEST_ROOT} 폴더가 없습니다.")
        raise SystemExit(0)

    print(f"\n📸 test_root = {TEST_ROOT}")
    print("=" * 70)

    for true_cat in CATS:
        cat_dir = os.path.join(TEST_ROOT, true_cat)
        if not os.path.isdir(cat_dir):
            print(f"⚠️ 폴더 없음: {cat_dir}")
            continue

        files = [f for f in os.listdir(cat_dir) if f.lower().endswith(EXTS)]
        files.sort()
        if PER_CLASS_LIMIT is not None:
            files = files[:PER_CLASS_LIMIT]

        print(f"\n🗂️ [{true_cat}] {len(files)}장 테스트 시작")

        for fn in files:
            img_path = os.path.join(cat_dir, fn)
            result = clf.classify(img_path)

            pred_final = result["category"]
            pred_img = result["image_result"]["category"]
            pred_text = result["text_result"]["category"]

            total += 1
            stats[true_cat]["total"] += 1
            img_stats[true_cat]["total"] += 1
            text_stats[true_cat]["total"] += 1

            is_correct = (pred_final == true_cat)
            if is_correct:
                correct_total += 1
                stats[true_cat]["correct"] += 1

            is_img_correct = (pred_img == true_cat)
            if is_img_correct:
                img_correct_total += 1
                img_stats[true_cat]["correct"] += 1
            
            is_text_correct = (pred_text == true_cat)
            if is_text_correct:
                text_correct_total += 1
                text_stats[true_cat]["correct"] += 1

            pred_key = pred_final if pred_final in CATS else "others"
            confusion[true_cat][pred_key] += 1

            # 실패한 경우만 출력
            if not is_correct:
                print(f"\n📄 {true_cat}/{fn}")
                print(
                    f"🏆 최종: {result['final_category_kr']} ({result['confidence']*100:.1f}%) "
                    f"{'✅' if is_correct else '❌'}"
                )
                print(
                    f"   - image: {result['image_result']['category']} "
                    f"({result['image_result']['confidence']*100:.1f}%)"
                    f"{' ✅' if is_img_correct else ''}"
                )
                print(f"   - text : {result['text_result']['category']} ({result['text_result']['confidence']*100:.1f}%)")
                print(f"   - ocr  : {result['ocr_text_preview'].replace(chr(10), ' ')[:80]}...")
                for log in result["logs"]:
                    print(f"   {log}")
                print("-" * 70)

    def pct(a, b):
        return (a / b * 100.0) if b else 0.0

    print("\n" + "=" * 70)
    print("📊 SUMMARY REPORT")
    print("=" * 70)

    print(f"✅ Final Accuracy: {correct_total}/{total} = {pct(correct_total, total):.2f}%")
    print(f"🖼️ Image-only Accuracy: {img_correct_total}/{total} = {pct(img_correct_total, total):.2f}%")
    print(f"📝 Text-only Accuracy: {text_correct_total}/{total} = {pct(text_correct_total, total):.2f}%")

    print("\n📌 Per-class Accuracy (Final):")
    for c in CATS:
        t = stats[c]["total"]
        k = stats[c]["correct"]
        print(f" - {c:10s}: {k:4d}/{t:4d} = {pct(k, t):6.2f}%")

    print("\n📌 Per-class Accuracy (Image-only):")
    for c in CATS:
        t = img_stats[c]["total"]
        k = img_stats[c]["correct"]
        print(f" - {c:10s}: {k:4d}/{t:4d} = {pct(k, t):6.2f}%")
    
    print("\n📌 Per-class Accuracy (Text-only):")
    for c in CATS:
        t = text_stats[c]["total"]
        k = text_stats[c]["correct"]
        print(f" - {c:10s}: {k:4d}/{t:4d} = {pct(k, t):6.2f}%")

    print("\n📌 Confusion Matrix (counts) [TRUE -> PRED]")
    header = "TRUE\\PRED".ljust(12) + "".join([p.rjust(12) for p in CATS])
    print(header)
    print("-" * len(header))
    for t in CATS:
        row = t.ljust(12)
        for p in CATS:
            row += str(confusion[t][p]).rjust(12)
        print(row)

    print(f"\n✅ Done. total tested = {total}")
