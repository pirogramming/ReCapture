from paddleocr import PaddleOCR
import os
import cv2
import numpy as np

class OCRService:
    """OCR 텍스트 추출 서비스 (전처리 포함)"""
    
    def __init__(self):
        """PaddleOCR 초기화"""
        print("🔧 PaddleOCR 초기화 중...")
        self.ocr = PaddleOCR(
            lang='korean', 
            use_angle_cls=True,
            det_db_thresh=0.3,      # 텍스트 감지 민감도 높임
            rec_batch_num=1         # 정확도 우선
        )
        print("✅ OCR 준비 완료!\n")
    
    def preprocess_image(self, image_path):
        """이미지 전처리 (강화 버전)"""
        img = cv2.imread(image_path)
        
        # 1. 리사이즈 (해상도 높이기)
        height, width = img.shape[:2]
        img = cv2.resize(img, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
        
        # 2. 그레이스케일
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. 가우시안 블러 (노이즈 제거)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 4. 샤프닝 (글자 선명하게)
        kernel = np.array([[-1,-1,-1],
                        [-1, 9,-1],
                        [-1,-1,-1]])
        sharpened = cv2.filter2D(blurred, -1, kernel)
        
        # 5. Adaptive Threshold (배경 제거 강화)
        binary = cv2.adaptiveThreshold(
            sharpened, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
        
        # 6. 반전 (검은 배경, 흰 글씨 → 흰 배경, 검은 글씨)
        inverted = cv2.bitwise_not(binary)
        
        # 7. 모폴로지 연산 (글자 굵기 조정)
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)
        
        # 저장
        temp_dir = os.path.dirname(image_path)
        temp_path = os.path.join(temp_dir, 'temp_preprocessed.jpg')
        cv2.imwrite(temp_path, morph)
        
        return temp_path
    
    def extract_text(self, image_path, use_preprocessing=True):
        """
        이미지에서 텍스트 추출
        
        Args:
            image_path (str): 이미지 파일 경로
            use_preprocessing (bool): 전처리 사용 여부 (기본 True)
            
        Returns:
            dict: {
                'full_text': 전체 텍스트,
                'lines': 라인별 텍스트 리스트,
                'confidences': 신뢰도 리스트
            }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
        
        # 전처리 여부 결정
        if use_preprocessing:
            print("🔄 이미지 전처리 중...")
            processed_path = self.preprocess_image(image_path)
        else:
            processed_path = image_path
        
        # OCR 실행
        print("📸 OCR 실행 중...")
        result = self.ocr.ocr(processed_path, cls=True)
        
        # 임시 파일 삭제
        if use_preprocessing and os.path.exists(processed_path):
            os.remove(processed_path)
        
        # 결과 파싱
        texts = []
        confidences = []
        
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]  # 텍스트
                confidence = line[1][1]  # 신뢰도
                
                if confidence > 0.5:  # 신뢰도 필터링
                    texts.append(text)
                    confidences.append(confidence)
        
        full_text = ' '.join(texts)
        
        return {
            'full_text': full_text,
            'lines': texts,
            'confidences': confidences
        }


# 테스트용 함수
if __name__ == "__main__":
    # 직접 실행했을 때만 동작
    ocr_service = OCRService()
    
    # 테스트 이미지 경로
    test_image = "test_data/non_info/image.png"
    
    if os.path.exists(test_image):
        print("=" * 50)
        print("전처리 없이 OCR")
        print("=" * 50)
        result1 = ocr_service.extract_text(test_image, use_preprocessing=False)
        print("📄 추출된 텍스트:")
        print(result1['full_text'])
        if result1['confidences']:
            print(f"📊 신뢰도 평균: {sum(result1['confidences']) / len(result1['confidences']):.2%}")
        
        print("\n" + "=" * 50)
        print("전처리 후 OCR")
        print("=" * 50)
        result2 = ocr_service.extract_text(test_image, use_preprocessing=True)
        print("📄 추출된 텍스트:")
        print(result2['full_text'])
        if result2['confidences']:
            print(f"📊 신뢰도 평균: {sum(result2['confidences']) / len(result2['confidences']):.2%}")
    else:
        print(f"⚠️ 테스트 이미지가 없습니다: {test_image}")