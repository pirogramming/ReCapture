# services/test_clustering.py

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.google_ocr_service import GoogleOCRService
from services.clustering_service import ClusteringService

class ClusteringTester:
    """
    🧪 2차 분류 (클러스터링) 테스트
    
    목적: 1차 분류 성능과 무관하게 "순수한 클러스터링 성능"만 평가
    """
    
    def __init__(self):
        print("="*70)
        print("🧪 클러스터링 테스트 시스템 초기화")
        print("="*70 + "\n")
        
        self.ocr = GoogleOCRService()
        self.clustering = ClusteringService(use_trained_models=True)  # 학습된 모델 사용!
    
    def test_category(self, category, test_dir='clustering_test_data'):
        """
        특정 카테고리 클러스터링 테스트
        
        Args:
            category: 'study_note' or 'info'
            test_dir: 테스트 데이터 루트 폴더
        """
        
        category_path = os.path.join(test_dir, category)
        
        if not os.path.exists(category_path):
            print(f"❌ 폴더 없음: {category_path}\n")
            print(f"💡 다음 구조로 준비하세요:")
            print(f"""
{test_dir}/
├── study_note/
│   ├── python_1.jpg    # 확실히 Python 관련
│   ├── python_2.jpg
│   ├── math_1.jpg      # 확실히 수학 관련
│   └── math_2.jpg
└── info/
    ├── shopping_1.jpg  # 확실히 쇼핑 관련
    └── travel_1.jpg    # 확실히 여행 관련
            """)
            return
        
        # 이미지 수집
        image_files = [
            f for f in os.listdir(category_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        if not image_files:
            print(f"❌ {category_path}에 이미지가 없습니다.\n")
            return
        
        print(f"📁 카테고리: {category}")
        print(f"📷 이미지 수: {len(image_files)}장\n")
        print("-"*70 + "\n")
        
        # ---------------------------------------------------
        # Step 1. OCR로 텍스트 추출
        # ---------------------------------------------------
        print("🔍 Step 1: OCR 텍스트 추출\n")
        
        image_data_list = []
        
        for i, filename in enumerate(image_files, 1):
            full_path = os.path.join(category_path, filename)
            
            try:
                ocr_result = self.ocr.extract_text(full_path)
                
                # 텍스트 추출
                if isinstance(ocr_result, dict):
                    text = ocr_result.get('full_text', '')
                else:
                    text = str(ocr_result)
                
                if len(text.strip()) < 5:
                    print(f"   [{i}/{len(image_files)}] {filename}: ⚠️ 텍스트 부족")
                    continue
                
                image_data_list.append({
                    'path': full_path,
                    'filename': filename,
                    'text': text
                })
                
                print(f"   [{i}/{len(image_files)}] {filename}: ✅ {len(text)}자")
            
            except Exception as e:
                print(f"   [{i}/{len(image_files)}] {filename}: ❌ 실패 ({e})")
        
        if not image_data_list:
            print("\n❌ OCR 성공한 이미지가 없습니다.\n")
            return
        
        print(f"\n✅ {len(image_data_list)}장 준비 완료\n")
        print("="*70 + "\n")
        
        # ---------------------------------------------------
        # Step 2. 클러스터링 실행 (SBERT vs KoELECTRA 비교)
        # ---------------------------------------------------
        print("🎯 Step 2: 클러스터링 실행 (1차 분류 학습 모델 활용 🔥)\n")
        
        result = self.clustering.cluster_images(
            image_data_list,
            min_cluster_size=2,  # 테스트용으로 낮춤
            min_samples=1,       # 더 완화
            image_weight=0.5     # 이미지:텍스트 = 50:50
        )
        
        if 'error' in result:
            print(f"❌ {result['error']}\n")
            return
        
        # 결과 출력 및 평가
        self._print_and_evaluate(result, category, 'TRAINED-ENSEMBLE')
    
    def _print_and_evaluate(self, result, category, text_model=''):
        """결과 출력 및 평가"""
        
        print("\n" + "="*70)
        print(f"📊 [{category}] 클러스터링 결과 ({text_model.upper()})")
        print("="*70 + "\n")
        
        clusters = result['clusters']
        noise = result['noise']
        
        if not clusters:
            print("❌ 클러스터를 생성하지 못했습니다.\n")
            print("💡 이미지 수를 늘리거나 min_cluster_size를 줄여보세요.\n")
            return
        
        # 각 클러스터 출력
        for cluster_id, data in sorted(clusters.items()):
            print(f"📦 그룹 {cluster_id + 1}")
            print(f"   🏷️  제안 이름: {data['suggested_name']}")
            print(f"   🔑 키워드: {', '.join(data['keywords'][:5])}")
            print(f"   📷 이미지 ({len(data['images'])}장):")
            
            for img_path in data['images']:
                filename = os.path.basename(img_path)
                print(f"      - {filename}")
            
            print()
        
        # 미분류
        if noise:
            print(f"🗑️  미분류 ({len(noise)}장)")
            for img_path in noise:
                filename = os.path.basename(img_path)
                print(f"   - {filename}")
            print()
        
        print("-"*70)
        print(f"📈 {result['summary']}")
        print("="*70 + "\n")
        
        # ---------------------------------------------------
        # 평가 가이드
        # ---------------------------------------------------
        print("💡 평가 포인트:")
        print("   ✅ 같은 주제 이미지들이 한 그룹에 모였나?")
        print("   ✅ 다른 주제 이미지들이 분리되었나?")
        print("   ✅ 제안된 이름이 적절한가?")
        print()
    
    def test_all(self):
        """study_note와 info 모두 테스트"""
        
        for category in ['study_note', 'info']:
            self.test_category(category)
            print("\n" + "🔄"*35 + "\n\n")

if __name__ == "__main__":
    tester = ClusteringTester()
    
    # 방법 1: 특정 카테고리만
    tester.test_category('study_note')
    
    # 방법 2: 전부 테스트
    # tester.test_all()