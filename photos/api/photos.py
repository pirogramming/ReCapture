# photos/api/photos.py
import os
from django.utils import timezone

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from rest_framework.pagination import PageNumberPagination

# from photos.models import Photo
from gallery.models import Photo
from photos.serializers.response import APIResponse
from photos.serializers.photo import PhotoSerializer, PhotoUpdateSerializer


def _user_photo_or_404(user, photo_id: int) -> Photo:
    return Photo.objects.get(id=photo_id, user=user, is_deleted=False)


@api_view(["GET"])
def list_photos(request):
    """
    (9) 사진 목록 조회
    - query optional: category, source
    """
    qs = Photo.objects.filter(
        user=request.user,
        is_deleted=False
    ).order_by("-created_at")

    category = request.query_params.get("category")
    source = request.query_params.get("source")

    if category:
        qs = qs.filter(category=category)
    if source:
        qs = qs.filter(source=source)

    paginator = PhotoPagination()
    page = paginator.paginate_queryset(qs, request)
    serializer = PhotoSerializer(page, many=True)

    return paginator.get_paginated_response(
        APIResponse.success({"items": serializer.data})
    )


@api_view(["GET"])
def get_photo_detail(request, photo_id: int):
    """
    (8) 사진 상세 조회
    """
    user = request.user
    try:
        photo = _user_photo_or_404(user, photo_id)
        return Response(APIResponse.success(PhotoSerializer(photo).data), status=status.HTTP_200_OK)
    except Photo.DoesNotExist:
        return Response(APIResponse.error("NOT_FOUND", "사진을 찾을 수 없습니다."), status=status.HTTP_404_NOT_FOUND)


@api_view(["PATCH"])
def update_photo(request, photo_id: int):
    """
    (10) 사진 메타 수정(카테고리/메모 등)
    """
    user = request.user
    try:
        photo = _user_photo_or_404(user, photo_id)
    except Photo.DoesNotExist:
        return Response(APIResponse.error("NOT_FOUND", "사진을 찾을 수 없습니다."), status=status.HTTP_404_NOT_FOUND)

    serializer = PhotoUpdateSerializer(photo, data=request.data, partial=True)
    if not serializer.is_valid():
        return Response(APIResponse.error("INVALID_REQUEST", "잘못된 요청입니다."), status=status.HTTP_400_BAD_REQUEST)

    serializer.save()
    return Response(APIResponse.success(PhotoSerializer(photo).data), status=status.HTTP_200_OK)


@api_view(["DELETE"])
def delete_photo(request, photo_id: int):
    """
    (12) 사진 삭제
    - soft delete + 파일 정리
    """
    user = request.user
    try:
        photo = _user_photo_or_404(user, photo_id)
    except Photo.DoesNotExist:
        return Response(APIResponse.error("NOT_FOUND", "사진을 찾을 수 없습니다."), status=status.HTTP_404_NOT_FOUND)

    # 파일 삭제 시도 (MEDIA_URL을 실제 경로로 바꿔서 삭제)
    # url이 "/media/photos/..." 같은 형태라면 MEDIA_ROOT 기준으로 변환 필요
    # 여기서는 "storage가 MEDIA_ROOT 아래"라는 전제에서 상대 경로로 변환
    # (환경에 따라 다를 수 있어, 안 맞으면 이 함수만 조정하면 됨)
    for u in [photo.url, photo.thumb_url]:
        try:
            if u and "/media/" in u:
                rel = u.split("/media/", 1)[1]
                abs_path = os.path.join("media", rel)  # 기본 media 폴더 가정
                if os.path.exists(abs_path):
                    os.remove(abs_path)
        except Exception:
            pass

    photo.is_deleted = True
    photo.deleted_at = timezone.now()
    photo.save(update_fields=["is_deleted", "deleted_at"])

    return Response(APIResponse.success({"deleted": True, "photoId": photo.id}), status=status.HTTP_200_OK)

class PhotoPagination(PageNumberPagination):
    page_size = 30