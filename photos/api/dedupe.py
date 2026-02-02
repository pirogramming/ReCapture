from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

# from photos.models import Photo
from gallery.models import Photo
from photos.serializers.response import APIResponse
from photos.serializers.photo import PhotoSerializer
from photos.services.deduplication_service import DeduplicationService


@api_view(["POST"])
def check_duplicates(request):
    """
    (11) 중복 제거
    - 완전히 동일한 이미지(SHA-256 동일)만 검사
    """
    user = request.user

    photo_id = request.data.get("photoId")
    file_hash = request.data.get("fileHash")

    if not photo_id and not file_hash:
        return Response(
            APIResponse.error(
                "INVALID_REQUEST",
                "photoId 또는 fileHash 중 하나는 필요합니다."
            ),
            status=status.HTTP_400_BAD_REQUEST,
        )

    exact = []

    if photo_id:
        try:
            photo = Photo.objects.get(
                id=photo_id,
                user=user,
                is_deleted=False,
            )
        except Photo.DoesNotExist:
            return Response(
                APIResponse.error("NOT_FOUND", "사진을 찾을 수 없습니다."),
                status=status.HTTP_404_NOT_FOUND,
            )

        exact = DeduplicationService.find_exact_duplicates(
            user,
            file_hash=photo.file_hash,
            exclude_photo_id=photo.id,
        )

    elif file_hash:
        exact = DeduplicationService.find_exact_duplicates(
            user,
            file_hash=file_hash,
        )

    return Response(
        APIResponse.success({
            "exact": [PhotoSerializer(p).data for p in exact],
        }),
        status=status.HTTP_200_OK,
    )
