import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json

class TextClassifier:
    """
    🧠 텍스트 분류기 (KoELECTRA-Base)
    학습된 모델을 불러와서 OCR로 추출된 텍스트의 카테고리를 예측합니다.
    (4개 카테고리: finance, study_note, info, others)
    """
    
    def __init__(self):
        print("🔧 텍스트 분류 모델 로딩 중...")
        
        # ---------------------------------------------------------
        # 1. 모델 경로 설정 (train_text.py에서 저장한 경로)
        # ---------------------------------------------------------
        # 프로젝트 루트 기준 경로
        self.model_path = 'classification/models/text_model_v1'
        
        if not os.path.exists(self.model_path):
            # classification 디렉토리 내부에서 실행되는 경우
            if os.path.exists('models/text_model_v1'):
                self.model_path = 'models/text_model_v1'
            else:
                raise FileNotFoundError(
                    f"❌ 모델 폴더가 없습니다: {self.model_path}\n"
                    f"먼저 training/train_text.py를 실행해서 모델을 학습시켜주세요!"
                )
        
        # ---------------------------------------------------------
        # 2. 디바이스 설정 (GPU 있으면 자동 사용)
        # ---------------------------------------------------------
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ---------------------------------------------------------
        # 3. 라벨 맵핑 로드 (★ 핵심: 하드코딩 제거 ★)
        # ---------------------------------------------------------
        # 학습할 때 저장된 순서 정보를 그대로 가져옵니다.
        label_map_path = os.path.join(self.model_path, 'label_map.json')
        
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
            # { "finance": 0 } -> { 0: "finance" } 형태로 뒤집기 (ID로 이름 찾기 위해)
            self.id_to_label = {v: k for k, v in label_map.items()}
            print(f"✅ 라벨 맵핑 로드 완료: {len(self.id_to_label)}개 카테고리")
        else:
            print("⚠️ 경고: label_map.json이 없습니다! (알파벳 순서로 임의 설정합니다)")
            # [수정] 4개 카테고리에 맞게 비상용 리스트 업데이트
            categories = sorted(['finance', 'info', 'others', 'study_note'])
            self.id_to_label = {i: cat for i, cat in enumerate(categories)}

        # ---------------------------------------------------------
        # 4. 모델 & 토크나이저 로드
        # ---------------------------------------------------------
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device) # 모델을 GPU로 이동
            self.model.eval()          # 평가 모드 (Dropout 끄기)
            print(f"✅ 모델 로드 성공 (Device: {self.device})")
        except Exception as e:
            raise RuntimeError(f"모델 로딩 실패: {e}")

        # [수정] 한글 매핑 (UI 출력용) - 4개 카테고리 반영
        self.category_kr = {
            'finance': '결제/금융',
            'study_note': '학습/노트',
            'info': '정보',
            'others': '기타(비정보)'
        }
    
    def predict(self, text):
        """
        텍스트를 입력받아 카테고리와 확신(Confidence)을 반환
        """
        # 1. 예외 처리: 텍스트가 너무 짧거나 없으면 'others' 취급
        if not text or len(text.strip()) < 2:
            return 'others', 0.0
        
        # 2. 토크나이징 (문장을 모델이 이해하는 숫자로 변환)
        inputs = self.tokenizer(
            text,
            return_tensors='pt',   # PyTorch Tensor로 반환
            max_length=256,        # 문맥 파악을 위해 넉넉하게
            truncation=True,       # 너무 길면 자르기
            padding=True
        )
        
        # 3. 입력 데이터를 GPU로 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 4. 예측 (Inference)
        with torch.no_grad(): # 평가 땐 기울기 계산 끔 (속도 향상)
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # 확률 계산 (Softmax)
            probs = torch.softmax(logits, dim=1)
            
            # 가장 높은 확률과 인덱스 찾기
            confidence, pred_idx = torch.max(probs, dim=1)
            
            pred_idx = pred_idx.item()
            confidence = confidence.item()
        
        # 5. 인덱스를 카테고리 이름으로 변환
        category = self.id_to_label[pred_idx]
        
        return category, confidence
    
    def predict_with_korean(self, text):
        """한글 카테고리 이름도 같이 반환하는 헬퍼 함수"""
        category, confidence = self.predict(text)
        category_kr = self.category_kr.get(category, category)
        return category, category_kr, confidence

# ====================================================
# 🧪 이 파일 자체 테스트용 코드
# ====================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 텍스트 분류기 단독 테스트 (4 Class)")
    print("=" * 60)
    
    try:
        classifier = TextClassifier()
        
        # 테스트 케이스 (새로운 카테고리에 맞춰 수정)
        test_samples = [
            ("스타벅스 아메리카노 4500원 결제완료", "finance"),
            ("미분방정식 3장 연습문제 풀이 수학", "study_note"),
            ("삼성전자 주주총회 소집 공고문", "info"),
            ("그냥 아무 의미 없는 텍스트 ㅋㅋㅋ", "others"),
            ("", "빈 텍스트")
        ]
        
        print("\n[예측 결과]")
        for text, label in test_samples:
            cat, cat_kr, conf = classifier.predict_with_korean(text)
            print(f"입력: {text[:15]}... | 예측: {cat_kr} ({cat}) | 확신: {conf*100:.1f}%")
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")