"""
사진 자동 분류 API
"""

from django.db import models
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status

from photos.models import Photo
from photos.serializers.response import APIResponse

# 1차 분류기 import
from classification.services.ensemble_classifier import EnsembleClassifier

# 전역 분류기 인스턴스
classifier = None

def get_classifier():
    """분류기 lazy loading"""
    global classifier
    if classifier is None:
        try:
            classifier = EnsembleClassifier()
            print("✅ 1차 분류기 로드 완료")
        except Exception as e:
            print(f"⚠️ 1차 분류기 로드 실패: {e}")
            return None
    return classifier


@api_view(["POST"])
@permission_classes([AllowAny])  # 세션 인증 사용
def classify_photos(request):
    """
    미분류 사진들을 자동으로 분류
    
    Request:
        - photo_ids (optional): 특정 사진들만 분류 [1, 2, 3]
        - 없으면 모든 미분류 사진 분류
    
    Response:
        - classified: 분류 성공한 사진 수
        - failed: 분류 실패한 사진 수
        - results: [{ id, filename, category, category_kr, confidence }]
    """
    # 세션 기반 로그인 체크
    if not request.user.is_authenticated:
        return Response(
            APIResponse.error("UNAUTHORIZED", "로그인이 필요합니다."),
            status=status.HTTP_401_UNAUTHORIZED,
        )
    
    user = request.user
    photo_ids = request.data.get('photo_ids', None)
    
    print(f"🔍 분류 요청: user={user.username}, photo_ids={photo_ids}")
    
    # 분류기 로드
    clf = get_classifier()
    if clf is None:
        return Response(
            APIResponse.error("CLASSIFIER_ERROR", "분류기를 로드할 수 없습니다."),
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    
    # 대상 사진 조회
    if photo_ids:
        photos = Photo.objects.filter(
            user=user,
            id__in=photo_ids,
            is_deleted=False,
        )
    else:
        # 미분류 사진만 (category가 None이거나 빈 문자열)
        photos = Photo.objects.filter(
            user=user,
            is_deleted=False,
        ).filter(
            models.Q(category__isnull=True) | models.Q(category='')
        )
    
    if not photos.exists():
        return Response(
            APIResponse.success({
                "classified": 0,
                "failed": 0,
                "results": [],
                "message": "분류할 사진이 없습니다."
            }),
            status=status.HTTP_200_OK,
        )
    
    # 분류 실행
    results = []
    classified_count = 0
    failed_count = 0
    
    for photo in photos:
        try:
            # 이미지 파일 경로 구하기
            import os
            from django.conf import settings
            
            # url에서 파일 경로 추출
            # url 형식: '/media/photos/2026/01/30/xxx.jpg'
            relative_path = photo.url.replace('/media/', '')
            image_path = os.path.join(settings.MEDIA_ROOT, relative_path)
            
            if not os.path.exists(image_path):
                failed_count += 1
                results.append({
                    "id": photo.id,
                    "filename": photo.filename,
                    "success": False,
                    "error": "파일을 찾을 수 없습니다."
                })
                continue
            
            # 분류 실행
            result = clf.classify(image_path)
            
            # DB 업데이트
            photo.category = result['category']
            photo.save(update_fields=['category'])
            
            classified_count += 1
            results.append({
                "id": photo.id,
                "filename": photo.filename,
                "success": True,
                "category": result['category'],
                "category_kr": result['category_kr'],
                "confidence": round(result['confidence'] * 100, 2),
            })
            
            print(f"✅ {photo.filename} → {result['category_kr']} ({result['confidence']*100:.1f}%)")
            
        except Exception as e:
            failed_count += 1
            results.append({
                "id": photo.id,
                "filename": photo.filename,
                "success": False,
                "error": str(e)
            })
            print(f"⚠️ 분류 실패 ({photo.filename}): {e}")
    
    return Response(
        APIResponse.success({
            "classified": classified_count,
            "failed": failed_count,
            "total": photos.count(),
            "results": results,
        }),
        status=status.HTTP_200_OK,
    )


@api_view(["GET"])
@permission_classes([AllowAny])
def get_unclassified_count(request):
    """미분류 사진 개수 조회"""
    if not request.user.is_authenticated:
        return Response(
            APIResponse.error("UNAUTHORIZED", "로그인이 필요합니다."),
            status=status.HTTP_401_UNAUTHORIZED,
        )
    
    user = request.user
    
    count = Photo.objects.filter(
        user=user,
        is_deleted=False,
    ).filter(
        models.Q(category__isnull=True) | models.Q(category='')
    ).count()
    
    return Response(
        APIResponse.success({"count": count}),
        status=status.HTTP_200_OK,
    )
