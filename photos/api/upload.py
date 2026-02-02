"""
주의:
이 앱은 TokenAuthentication + IsAuthenticated 전제를 기반으로 동작한다.
accounts 앱에서 토큰 발급 API가 반드시 선행되어야 한다.
"""

import os

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

# from photos.models import Photo
from gallery.models import Photo
from photos.serializers.response import APIResponse
from photos.serializers.photo import PhotoSerializer
from photos.services.upload_service import save_uploaded_file
from photos.services.deduplication_service import DeduplicationService

@api_view(["POST"])
def upload_photos(request):
    """
    (5) 직접 업로드
    - SHA-256 기준 완전히 동일한 파일만 중복 처리
    """
    user = request.user
    files = request.FILES.getlist("files")

    if not files:
        return Response(
            APIResponse.error("NO_FILES", "업로드할 파일이 없습니다."),
            status=status.HTTP_400_BAD_REQUEST,
        )

    created = []
    duplicates = []

    for f in files:
        meta = save_uploaded_file(user, f)

        # 1️⃣ Exact 중복 사전 체크
        existing = Photo.objects.filter(
            user=user,
            file_hash=meta["file_hash"],
            is_deleted=False,
        ).first()

        if existing:
            # 방금 저장한 파일/썸네일 정리
            for path in (meta.get("_final_path"), meta.get("_thumb_path")):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

            duplicates.append({
                "filename": meta["filename"],
                "existingPhotoId": existing.id,
            })
            continue

        # 2️⃣ Photo 생성
        photo = Photo.objects.create(
            user=user,
            filename=meta["filename"],
            url=meta["url"],
            thumb_url=meta["thumb_url"],
            file_size=meta["file_size"],
            file_hash=meta["file_hash"],
            width=meta["width"],
            height=meta["height"],
            taken_at=meta["taken_at"],
            source="UPLOAD",
        )

        # 3️⃣ 생성 직후 exact duplicate 재확인 + 마킹
        DeduplicationService.check_exact_duplicate_and_mark(
            user,
            photo=photo,
        )

        created.append(PhotoSerializer(photo).data)

    return Response(
        APIResponse.success({
            "created": created,
            "duplicates": duplicates,
        }),
        status=status.HTTP_200_OK,
    )
