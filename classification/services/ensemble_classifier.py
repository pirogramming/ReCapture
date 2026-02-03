import os
import sys

# 프로젝트 루트 경로 설정 (services 폴더의 상위 폴더를 참조하기 위함)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# [수정 1] 이제 옛날 OCRService(Paddle) 대신 GoogleOCRService를 씁니다.
from services.google_ocr_service import GoogleOCRService
from services.image_classifier import ImageClassifier
from services.text_classifier import TextClassifier

class EnsembleClassifier:
    """
    🚀 스마트 통합 분류기 (4개 카테고리 최적화 버전)
    카테고리: finance(금융), study_note(학습), info(정보), others(기타)
    전략: Vision First (속도) -> Text Confirmation (정확도)
    """
    
    def __init__(self):
        print(" 통합 분류 시스템 초기화 중...\n")
        
        try:
            self.ocr = GoogleOCRService()
        except Exception as e:
            print(f" Google OCR 초기화 실패: {e}")
            print("   -> google_credentials.json 경로를 확인하세요.")
            # OCR 실패 시 비전 모델만이라도 돌리기 위해 None 처리 가능하지만, 
            # 여기서는 에러를 보고 고치는 게 나으므로 일단 진행
            
        self.image_clf = ImageClassifier()
        self.text_clf = TextClassifier()
        
        # 🚦 OCR 검문이 필요한 카테고리 (Trigger)
        self.ocr_triggers = ['finance', 'info', 'study_note']
        
        print(" 모든 모델 로드 완료!\n")
    
    def classify(self, image_path):
        """
        Args:
            image_path (str): 이미지 경로
        Returns:
            dict: 최종 분류 결과 (한국어 포함)
        """
        logs = [] # 디버깅용 로그
        
        # ---------------------------------------------------
        # Step 1. 비전 모델 가동 (속도 빠름 ⚡)
        # ---------------------------------------------------
        img_cat, img_conf = self.image_clf.predict(image_path)
        logs.append(f" 비전 예측: {img_cat} ({img_conf*100:.1f}%)")
        
        # 기본값 설정
        final_cat = img_cat
        final_conf = img_conf
        ocr_text = ""
        text_cat = None
        text_conf = 0.0

        # ---------------------------------------------------
        # Step 2. OCR 발동 조건 체크 (Smart Trigger 🚦)
        # ---------------------------------------------------
        is_document = img_cat in self.ocr_triggers
        is_uncertain = img_conf < 0.6
        
        if is_document or is_uncertain:
            logs.append(f" OCR 검문 시작 (사유: {img_cat} 타입 or 확신 부족)")
            
            try:
                # OCR 실행 (Google Vision)
                ocr_result = self.ocr.extract_text(image_path)
                
                # 딕셔너리/문자열 처리
                if isinstance(ocr_result, dict):
                    ocr_text = ocr_result.get('full_text', '')
                else:
                    ocr_text = str(ocr_result)

                if len(ocr_text.strip()) < 5:
                    logs.append(" 글자 없음. 이미지 결과 유지.")
                
                else:
                    # -----------------------------------------------
                    # Step 3. 텍스트 모델 가동 (정확도 높음 🧠)
                    # -----------------------------------------------
                    text_cat, text_conf = self.text_clf.predict(ocr_text)
                    logs.append(f" 텍스트 예측: {text_cat} ({text_conf*100:.1f}%)")
                    
                    # -----------------------------------------------
                    # Step 4. 최종 판결 (Conflict Resolution)
                    # -----------------------------------------------
                    if text_conf > 0.8:
                        final_cat = text_cat
                        final_conf = text_conf
                        logs.append(" 텍스트 확신 높음 -> 결과 덮어쓰기")
                    
                    elif img_cat != text_cat:
                        if text_cat != 'others':
                            final_cat = text_cat
                            final_conf = text_conf
                            logs.append(" 의견 불일치 -> 텍스트 결과 우선")
                        else:
                            logs.append(" 텍스트가 '기타'라고 함 -> 이미지 결과 유지")
                    
                    else:
                        final_conf = min((img_conf + text_conf) / 2 + 0.1, 0.99)
                        logs.append(" 의견 일치 -> 확신도 증가")

            except Exception as e:
                logs.append(f" OCR 실패: {e}")
        
        else:
            logs.append(" 시각 정보 확실함 (OCR 생략)")

        # ---------------------------------------------------
        # Step 5. 한국어 변환 및 반환
        # ---------------------------------------------------
        category_kr_map = {
            'finance': '결제/금융',
            'study_note': '학습/노트',
            'info': '정보',
            'others': '기타(비정보)'
        }
        
        return {
            'category': final_cat,
            'final_category_kr': category_kr_map.get(final_cat, '알수없음'),
            'confidence': round(final_conf, 4),
            'image_result': {'category': img_cat, 'confidence': img_conf},
            'text_result': {'category': text_cat, 'confidence': text_conf},
            'ocr_text': ocr_text[:100],
            'logs': logs
        }

if __name__ == "__main__":
    classifier = EnsembleClassifier()
    
    # test_data 폴더에서 테스트
    test_dir = "test_data"
    
    if not os.path.exists(test_dir):
        print("\n test_data 폴더가 없습니다.")
    else:
        # test_data의 모든 이미지 테스트
        test_files = [f for f in os.listdir(test_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not test_files:
            print("\n test_data에 이미지가 없습니다.")
        else:
            print(f"\n {len(test_files)}개 이미지 테스트\n")
            print("="*70)
            
            for filename in test_files[:]:  # 5장만
                full_path = os.path.join(test_dir, filename)
                result = classifier.classify(full_path)
                
                print(f"\n {filename}")
                print(f"🏆 결과: {result['final_category_kr']} ({result['confidence']*100:.1f}%)")
                for log in result['logs']:
                    print(f"   {log}")
                print("-"*70)