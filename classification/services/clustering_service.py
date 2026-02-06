# services/clustering_service.py

import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

class ClusteringService:
    """
    📊 이미지 클러스터링 서비스 (순수 로직)
    - 텍스트 임베딩
    - HDBSCAN 클러스터링
    - 키워드 추출
    """
    
    def __init__(self, use_image_features=True, use_trained_models=True):
        print("🔧 클러스터링 모델 로드 중...")
        
        self.use_trained_models = use_trained_models
        
        if use_trained_models:
            print("🎯 1차 분류 학습된 모델 사용 (고성능)")
            
            # 학습된 KoELECTRA 모델 로드
            from transformers import AutoTokenizer, AutoModel
            print("📥 학습된 KoELECTRA 모델 로드 중...")
            model_path = 'models/text_model_v1'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"❌ 학습된 텍스트 모델이 없습니다: {model_path}")
            
            self.koelectra_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.koelectra_model = AutoModel.from_pretrained(model_path)
            self.koelectra_model.eval()
            print("✅ 학습된 KoELECTRA 준비 완료")
            
            # 학습된 EfficientNet 모델 로드
            print("📥 학습된 EfficientNet 모델 로드 중...")
            efficientnet_path = 'models/efficientnet_v1.pth'
            if not os.path.exists(efficientnet_path):
                raise FileNotFoundError(f"❌ 학습된 이미지 모델이 없습니다: {efficientnet_path}")
            
            # EfficientNet feature extractor (분류기가 아닌 특징 추출기로 변경)
            import timm
            self.image_model = timm.create_model(
                'efficientnet_b0',
                pretrained=False,
                num_classes=0  # feature extractor mode
            )
            # 학습된 가중치 로드하되, classifier layer는 제외하고 feature만 추출
            state_dict = torch.load(efficientnet_path, map_location='cpu')
            # classifier 레이어 제거
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
            self.image_model.load_state_dict(state_dict, strict=False)
            self.image_model.eval()
            
            from torchvision import transforms
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            print("✅ 학습된 EfficientNet 준비 완료")
            
        else:
            print("🎯 일반 모델 사용")
            
            # 텍스트 임베딩 모델들 (비교 테스트용)
            print("📥 SBERT 로드 중...")
            self.sbert_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            print("✅ SBERT 준비 완료")
            
            print("📥 KoELECTRA 로드 중...")
            from transformers import AutoTokenizer, AutoModel
            self.koelectra_tokenizer = AutoTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
            self.koelectra_model = AutoModel.from_pretrained('monologg/koelectra-base-v3-discriminator')
            self.koelectra_model.eval()
            print("✅ KoELECTRA 준비 완료")
            
            # 이미지 특징 추출 모델 (옵션)
            self.use_image_features = use_image_features
            if use_image_features:
                print("🔧 이미지 특징 추출 모델 로드 중...")
                import timm
                self.image_model = timm.create_model(
                    'efficientnet_b0',
                    pretrained=True,
                    num_classes=0  # feature extractor mode
                )
                self.image_model.eval()
                
                from torchvision import transforms
                self.image_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                print("✅ 이미지 특징 추출 모델 준비 완료")
        
        print()
    
    def cluster_images(self, image_data_list, min_cluster_size=3, min_samples=2, image_weight=0.5, text_model='sbert'):
        """
        이미지 텍스트들을 클러스터링 (이미지 특징 + 텍스트 임베딩)
        
        Args:
            image_data_list: [{'path': '경로', 'text': 'OCR텍스트'}, ...]
            min_cluster_size: 최소 클러스터 크기 (기본 3)
            min_samples: 최소 샘플 수 (기본 2)
            image_weight: 이미지 특징 가중치 (0.0~1.0, 기본 0.5)
                         0.0 = 텍스트만, 1.0 = 이미지만, 0.5 = 동일 비중
            text_model: 텍스트 임베딩 모델 ('sbert' 또는 'koelectra')
        
        Returns:
            {
                'clusters': {cluster_id: {'images': [...], 'keywords': [...], ...}},
                'noise': [...],
                'summary': '...'
            }
        """
        
        if len(image_data_list) < min_cluster_size:
            return {
                'error': f'이미지 부족 (최소 {min_cluster_size}장 필요)',
                'clusters': {},
                'noise': [item['path'] for item in image_data_list]
            }
        
        # Step 1. 텍스트 전처리 및 임베딩
        texts = [self._preprocess_text(item['text']) for item in image_data_list]
        
        # 텍스트 모델 선택
        if self.use_trained_models:
            print(f"🔤 텍스트 임베딩: 학습된 KoELECTRA 사용")
            text_embeddings = self._encode_with_koelectra(texts)
        elif text_model == 'koelectra':
            print(f"🔤 텍스트 임베딩: KoELECTRA 사용")
            text_embeddings = self._encode_with_koelectra(texts)
        else:  # sbert
            print(f"🔤 텍스트 임베딩: SBERT 사용")
            text_embeddings = self.sbert_model.encode(texts, show_progress_bar=False)
        
        # Step 1-2. 이미지 특징 추출
        if self.use_trained_models or (hasattr(self, 'use_image_features') and self.use_image_features):
            image_embeddings = self._extract_image_features([item['path'] for item in image_data_list])
            
            # 정규화 (0~1 범위로 스케일링)
            from sklearn.preprocessing import StandardScaler
            text_norm = StandardScaler().fit_transform(text_embeddings)
            image_norm = StandardScaler().fit_transform(image_embeddings)
            
            # 차원이 다르므로 가중 평균 대신 연결(concatenate) 후 가중치 적용
            # image_weight가 높으면 이미지 특징에 더 큰 가중치
            text_weighted = text_norm * (1 - image_weight)
            image_weighted = image_norm * image_weight
            
            # 연결 (concatenate)
            embeddings = np.concatenate([text_weighted, image_weighted], axis=1)
        else:
            embeddings = text_embeddings
        
        # Step 2. HDBSCAN 클러스터링
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        cluster_labels = clusterer.fit_predict(embeddings)
        

        # Step 3. 결과 정리
        clusters = {}
        noise_images = []
        
        for item, label in zip(image_data_list, cluster_labels):
            if label == -1:
                noise_images.append(item['path'])
                continue
            
            if label not in clusters:
                clusters[label] = {'images': [], 'texts': []}
            
            clusters[label]['images'].append(item['path'])
            clusters[label]['texts'].append(item['text'])
        
        # Step 4. 각 클러스터 키워드 추출
        for cluster_id, data in clusters.items():
            keywords = self._extract_keywords(data['texts'])
            clusters[cluster_id]['keywords'] = keywords
            clusters[cluster_id]['suggested_name'] = self._suggest_name(keywords)
            del clusters[cluster_id]['texts']  # 메모리 절약
        
        return {
            'clusters': clusters,
            'noise': noise_images,
            'summary': f"총 {len(clusters)}개 그룹 (노이즈 {len(noise_images)}장)"
        }
    
    def _encode_with_koelectra(self, texts):
        """KoELECTRA로 텍스트 임베딩 (CLS 토큰)"""
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # 토큰화
                inputs = self.koelectra_tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                
                # 임베딩 추출 (CLS 토큰 사용)
                outputs = self.koelectra_model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                embeddings.append(cls_embedding)
        
        return np.array(embeddings)
    
    def _extract_image_features(self, image_paths):
        """이미지에서 시각적 특징 추출 (EfficientNet)"""
        features = []
        
        with torch.no_grad():
            for img_path in image_paths:
                try:
                    # 이미지 로드 및 전처리
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.image_transform(img).unsqueeze(0)
                    
                    # 특징 추출 (1280차원 벡터)
                    feature = self.image_model(img_tensor).squeeze().numpy()
                    features.append(feature)
                except Exception as e:
                    print(f"   ⚠️ 이미지 특징 추출 실패 ({os.path.basename(img_path)}): {e}")
                    # 에러 시 zero vector
                    features.append(np.zeros(1280))  # EfficientNet-B0 feature dim
        
        return np.array(features)
    
    def _preprocess_text(self, text):
        """텍스트 전처리: 의미있는 단어만 추출"""
        import re
        # 특수문자 제거하되 한글, 영문, 숫자, 공백만 남김
        text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
        # 연속 공백 제거
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _extract_keywords(self, texts, top_n=10):
        """TF-IDF 키워드 추출 (의미있는 단어만)"""
        if not texts:
            return []
        
        # 대폭 확장된 불용어 리스트 (영어 상위 100개 + 한국어)
        stop_words = [
            # 한국어 불용어
            '것', '수', '등', '및', '이', '그', '저', '를', '을', '가', '은', '는', '의', '에', '로', '으로',
            '과', '와', '도', '만', '라', '에서', '으로써', '하다', '되다', '있다', '없다',
            # 영어 고빈도 불용어
            'the', 'a', 'an', 'and', 'or', 'to', 'in', 'of', 'for', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'it',
            'he', 'she', 'they', 'we', 'you', 'me', 'him', 'her', 'them', 'us', 'my', 'your',
            'his', 'its', 'our', 'their', 'at', 'by', 'from', 'with', 'about', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out',
            'on', 'off', 'over', 'under', 'again', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'just', 'but', 'if', 'because', 'while', 'who', 'what', 'which'
        ]
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words=stop_words,
                ngram_range=(1, 3),  # 3-gram까지 확장 (고유명사 캡처)
                max_df=0.7,  # 70% 이상 문서에 나오면 제외 (너무 흔한 단어)
                min_df=1,
                token_pattern=r'\b[가-힣a-zA-Z]{3,}\b'  # 3글자 이상으로 강화
            )
            
            combined = ' '.join(texts)
            tfidf_matrix = vectorizer.fit_transform([combined])
            
            features = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            top_idx = scores.argsort()[-top_n:][::-1]
            return [features[i] for i in top_idx]
        
        except:
            # 폴백: 단순 빈도 (2글자 이상 단어만)
            import re
            words = [w for w in ' '.join(texts).split() if len(re.sub(r'[^가-힣a-zA-Z]', '', w)) >= 2]
            return [word for word, _ in Counter(words).most_common(top_n)]
    
    def _suggest_name(self, keywords):
        """클러스터 이름 제안"""
        if not keywords:
            return "미분류"
        return f"{keywords[0]}/{keywords[1]}" if len(keywords) >= 2 else keywords[0]