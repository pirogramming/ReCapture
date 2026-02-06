import torch
import timm
from PIL import Image
from torchvision import transforms
import os

class ImageClassifier:
    """이미지 분류기 (EfficientNet-B0) - 4 Class Version"""
    
    def __init__(self):
        """모델 로드"""
        print("🔧 이미지 분류 모델 로딩 중...")
        
        # ---------------------------------------------------------
        # 1. 모델 경로 (train_image.py에서 저장한 경로와 일치해야 함)
        # ---------------------------------------------------------
        # 프로젝트 루트 기준 경로
        model_path = 'classification/models/efficientnet_v1.pth'
        
        if not os.path.exists(model_path):
            # classification 디렉토리 내부에서 실행되는 경우
            if os.path.exists('models/efficientnet_v1.pth'):
                model_path = 'models/efficientnet_v1.pth'
            else:
                raise FileNotFoundError(
                    f"❌ 모델 파일이 없습니다: {model_path}\n"
                    f"먼저 training/train_image.py를 실행해서 모델을 학습하세요!"
                )
        
        print(f"   📂 모델 파일: {model_path}")

        # ---------------------------------------------------------
        # 2. 모델 생성 (★수정: num_classes=4)
        # ---------------------------------------------------------
        self.model = timm.create_model(
            'efficientnet_b0',
            pretrained=False,  # 추론 때는 False
            num_classes=4      # [finance, info, others, study_note]
        )
        
        # ---------------------------------------------------------
        # 3. 가중치 로드
        # ---------------------------------------------------------
        try:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()  # 평가 모드
        except Exception as e:
            raise RuntimeError(f"모델 가중치 로드 실패 (혹시 카테고리 수가 안 맞나요?): {e}")
        
        # 전처리 (학습 시와 동일)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # ---------------------------------------------------------
        # 4. 카테고리 이름 (★수정: 알파벳 순서 필수!)
        # ---------------------------------------------------------
        # train_image.py는 폴더를 알파벳순으로 읽습니다. 여기서도 똑같이 맞춰야 합니다.
        self.categories = sorted([
            'finance',      # 결제/금융
            'info',         # 정보
            'others',       # 기타(비정보)
            'study_note'    # 학습/노트
        ])
        
        # ---------------------------------------------------------
        # 5. 한글 매핑 (★수정: 4개 체제 반영)
        # ---------------------------------------------------------
        self.category_kr = {
            'finance': '결제/금융',
            'info': '정보',
            'others': '기타(비정보)',
            'study_note': '학습/노트'
        }
        
        print(f"✅ 모델 준비 완료! (카테고리: {self.categories})\n")
    
    def predict(self, image_path):
        """
        이미지 분류 예측
        Returns: (카테고리, 확률)
        """
        # 이미지 로드
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
            image = Image.open(image_path).convert('RGB')
        else:
            # Django UploadedFile 객체인 경우
            image = Image.open(image_path).convert('RGB')
        
        # 전처리
        image_tensor = self.transform(image).unsqueeze(0)
        
        # 예측
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = torch.argmax(probs).item()
            confidence = probs[0][pred_idx].item()
        
        category = self.categories[pred_idx]
        
        return category, confidence
    
    def predict_with_korean(self, image_path):
        """한글 카테고리와 함께 반환"""
        category, confidence = self.predict(image_path)
        category_kr = self.category_kr.get(category, category)
        return category, category_kr, confidence

# ========================================
# 테스트 코드
# ========================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 이미지 분류 테스트 (4 Class)")
    print("=" * 60)
    
    try:
        classifier = ImageClassifier()
        
        # 테스트 이미지 경로 (경로 확인 필요)
        test_dir = 'train_data' # 일단 train_data에 있는 걸로 테스트
        
        if os.path.exists(test_dir):
            print(f"📸 '{test_dir}' 폴더의 이미지로 테스트합니다.\n")
            
            # 각 카테고리별로 1장씩만 뽑아서 테스트
            for cat in classifier.categories:
                cat_dir = os.path.join(test_dir, cat)
                if os.path.isdir(cat_dir):
                    files = [f for f in os.listdir(cat_dir) if f.lower().endswith('.jpg')]
                    if files:
                        img_path = os.path.join(cat_dir, files[0])
                        cat_res, cat_kr, conf = classifier.predict_with_korean(img_path)
                        print(f"[{cat}] 폴더 파일 -> 예측: {cat_kr} ({conf*100:.1f}%)")
        else:
            print("⚠️ 테스트할 폴더가 없습니다.")

    except Exception as e:
        print(f"\n❌ 오류: {e}")