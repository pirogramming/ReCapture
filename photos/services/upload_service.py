# photos/services/upload_service.py

import os
import uuid
import hashlib
from datetime import datetime
from typing import Dict

from django.conf import settings
from django.core.files.uploadedfile import UploadedFile

from PIL import Image, ExifTags
import imagehash


# =========================
# Path Utils
# =========================

def _ensure_dir(path: str):
    """디렉토리 없으면 생성"""
    os.makedirs(path, exist_ok=True)


def _user_storage_paths(user_id: int) -> Dict[str, str]:
    """
    사용자별 저장 경로 반환
    """
    base = settings.MEDIA_ROOT
    return {
        "temp": os.path.join(base, "temp", str(user_id)),
        "photo": os.path.join(base, "photos", str(user_id)),
        "thumb": os.path.join(base, "thumbnails", str(user_id)),
    }


# =========================
# Hash Utils
# =========================

def _calculate_sha256(file_path: str) -> str:
    """SHA-256 해시 계산 (Exact duplicate)"""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _calculate_image_hashes(image: Image.Image) -> Dict[str, str]:
    """pHash / dHash / aHash 계산"""
    return {
        "phash": str(imagehash.phash(image)),
        "dhash": str(imagehash.dhash(image)),
        "ahash": str(imagehash.average_hash(image)),
    }


# =========================
# Image Utils
# =========================

def _extract_exif_taken_at(image: Image.Image):
    """EXIF 촬영 시간 추출"""
    try:
        exif = image._getexif()
        if not exif:
            return None

        for tag, value in exif.items():
            tag_name = ExifTags.TAGS.get(tag)
            if tag_name == "DateTimeOriginal":
                return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None

    return None


def _create_thumbnail(image: Image.Image, size=(300, 300)) -> Image.Image:
    """썸네일 생성"""
    thumb = image.copy()
    thumb.thumbnail(size)
    return thumb


# =========================
# Main Service
# =========================

def save_uploaded_file(user, uploaded_file: UploadedFile) -> Dict:
    """
    업로드된 이미지 파일 저장 및 메타데이터 추출

    return:
    {
        filename,
        url,
        thumb_url,
        file_size,
        width,
        height,
        taken_at,
        file_hash,
        phash,
        dhash,
        ahash,
    }
    """

    user_id = user.id
    paths = _user_storage_paths(user_id)

    for p in paths.values():
        _ensure_dir(p)

    # 파일명 정리 (충돌 방지)
    original_name = uploaded_file.name
    ext = os.path.splitext(original_name)[1].lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"

    temp_path = os.path.join(paths["temp"], unique_name)
    final_path = os.path.join(paths["photo"], unique_name)
    thumb_path = os.path.join(paths["thumb"], unique_name)

    # 임시 저장
    with open(temp_path, "wb+") as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)

    # 이미지 로드
    image = Image.open(temp_path)
    image = image.convert("RGB")  # 포맷 통일

    width, height = image.size
    file_size = uploaded_file.size

    # 해시 계산
    file_hash = _calculate_sha256(temp_path)
    image_hashes = _calculate_image_hashes(image)

    # 메타데이터
    taken_at = _extract_exif_taken_at(image)

    # 원본 저장
    image.save(final_path, format="JPEG", quality=95)

    # 썸네일 생성
    thumbnail = _create_thumbnail(image)
    thumbnail.save(thumb_path, format="JPEG", quality=85)

    # temp 파일 삭제
    try:
        os.remove(temp_path)
    except OSError:
        pass

    # URL 구성
    photo_url = f"{settings.MEDIA_URL}photos/{user_id}/{unique_name}"
    thumb_url = f"{settings.MEDIA_URL}thumbnails/{user_id}/{unique_name}"

    return {
        "filename": original_name,
        "url": photo_url,
        "thumb_url": thumb_url,
        "file_size": file_size,
        "width": width,
        "height": height,
        "taken_at": taken_at,
        "file_hash": file_hash,
        "phash": image_hashes["phash"],
        "dhash": image_hashes["dhash"],
        "ahash": image_hashes["ahash"],

        
        # 내부 처리용(중복 시 파일 정리 등). 외부 응답에는 쓰지 않아도 됨.
        "_final_path": final_path,
        "_thumb_path": thumb_path,
    }
