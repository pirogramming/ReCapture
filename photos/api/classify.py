import json
from django.http import JsonResponse
from django.views.decorators.http import require_POST, require_GET
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from photos.models import Photo

# 1. 앙상블 분류기 가져오기
# (프로젝트 구조에 따라 import 경로는 'classification.services...' 일 수 있습니다)
from classification.services.ensemble_classifier import EnsembleClassifier

# 🚀 분류기 인스턴스 초기화 (서버 실행 시 1회만 로드하여 속도 향상)
# try-except로 감싸서 모델 파일이 없어도 서버가 죽지 않게 방어
try:
    classifier = EnsembleClassifier()
    print("✅ EnsembleClassifier 로드 완료 (photos/api/classify.py)")
except Exception as e:
    print(f"⚠️ 분류기 로드 실패: {e}")
    classifier = None

@csrf_exempt
@require_POST
def classify_photos(request):
    """
    [POST] /api/v1/photos/classify/
    미분류('unclassified' 또는 NULL) 사진들을 자동으로 분류합니다.
    """
    if not classifier:
        return JsonResponse({'status': 'error', 'message': '서버에 분류기가 로드되지 않았습니다.'}, status=500)

    try:
        # 1. 분류 대상 가져오기 (카테고리가 없거나 'unclassified'인 사진)
        # category__isnull=True: 카테고리가 비어있는 것
        # category='unclassified': 명시적으로 미분류로 된 것
        target_photos = Photo.objects.filter(category__isnull=True) | Photo.objects.filter(category='unclassified')
        
        # 대상이 없으면 바로 리턴
        if not target_photos.exists():
            return JsonResponse({'status': 'success', 'classified_count': 0, 'message': '분류할 사진이 없습니다.'})

        success_count = 0
        results = []

        for photo in target_photos:
            try:
                # 2. 이미지 경로 확보
                image_path = photo.image.path
                
                # 3. 앙상블 분류기 실행 (제공해주신 classify 메서드 호출)
                # 반환값 예시: {'category': 'finance', 'confidence': 0.98, ...}
                classification_result = classifier.classify(image_path)
                
                final_category = classification_result.get('category')
                
                # 4. 결과가 유효하면 DB 업데이트
                if final_category:
                    photo.category = final_category
                    
                    # (선택사항) OCR 텍스트가 있다면 DB에 같이 저장
                    # Photo 모델에 ocr_text 필드가 있다고 가정
                    if hasattr(photo, 'ocr_text'):
                        # ensemble_classifier.py의 _get_ocr_text_cached 로직에 의해
                        # 캐시된 텍스트를 가져오거나 result logs에서 유추해야 할 수 있음.
                        # 여기서는 단순 분류만 처리합니다.
                        pass

                    photo.save()
                    success_count += 1
                    
                    results.append({
                        'id': photo.id,
                        'category': final_category,
                        'category_kr': classification_result.get('final_category_kr', '')
                    })
                    
            except Exception as e:
                print(f"⚠️ 사진(ID {photo.id}) 분류 중 에러: {e}")
                continue

        return JsonResponse({
            'status': 'success',
            'classified_count': success_count,
            'results': results
        })

    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@require_GET
def get_unclassified_count(request):
    """
    [GET] /api/v1/photos/classify/count/
    미분류 사진 개수를 반환합니다. (배지 표시용)
    """
    count = Photo.objects.filter(category__isnull=True).count() + \
            Photo.objects.filter(category='unclassified').count()
    return JsonResponse({'count': count})