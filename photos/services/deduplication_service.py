from typing import List, Optional
from photos.models import Photo


class DeduplicationService:
    """
    완전히 동일한 이미지(SHA-256 동일)만 중복으로 판단
    """

    @staticmethod
    def find_exact_duplicates(
        user,
        *,
        file_hash: str,
        exclude_photo_id: Optional[int] = None,
    ) -> List[Photo]:
        """
        같은 user + 같은 file_hash를 가진 사진 조회
        """
        qs = Photo.objects.filter(
            user=user,
            file_hash=file_hash,
            is_deleted=False,
        )

        if exclude_photo_id is not None:
            qs = qs.exclude(id=exclude_photo_id)

        return list(qs.order_by("created_at"))

    @staticmethod
    def mark_exact_duplicate(photo: Photo, original: Photo) -> None:
        """
        photo를 original의 exact duplicate로 표시
        """
        photo.duplicate_of = original
        photo.save(update_fields=["duplicate_of"])

    @staticmethod
    def check_exact_duplicate_and_mark(
        user,
        *,
        photo: Photo,
    ) -> Optional[Photo]:
        """
        photo 기준으로 exact duplicate 검사 후,
        있으면 DB에 duplicate_of 저장
        """
        exact = DeduplicationService.find_exact_duplicates(
            user,
            file_hash=photo.file_hash,
            exclude_photo_id=photo.id,
        )

        if exact:
            original = exact[0]
            DeduplicationService.mark_exact_duplicate(photo, original)
            return original

        return None
