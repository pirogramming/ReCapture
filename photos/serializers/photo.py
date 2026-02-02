# photos/serializers/photo.py
from rest_framework import serializers
# from photos.models import Photo
from gallery.models import Photo

class ImportGoogleRequestSerializer(serializers.Serializer):
    """구글 Import 요청"""
    folderId = serializers.CharField(required=False, allow_null=True)
    dedupe = serializers.BooleanField(default=True)

class ImportJobStatusResponseSerializer(serializers.Serializer):
    """Import Job 상태 응답"""
    jobId = serializers.CharField()
    status = serializers.CharField()
    progress = serializers.DictField()

class ImportJobStartResponseSerializer(serializers.Serializer):
    """Import Job 시작 응답"""
    jobId = serializers.CharField()
    status = serializers.CharField()

class PhotoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Photo
        fields = [
            "id",
            "filename",
            "url",
            "thumb_url",
            "file_size",
            "file_hash",
            "phash",
            "dhash",
            "ahash",
            "category",
            "sub_category",
            "source",
            "google_id",
            "memo",
            "width",
            "height",
            "taken_at",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id", "url", "thumb_url", "file_size",
            "file_hash", "phash", "dhash", "ahash",
            "source", "google_id",
            "width", "height", "taken_at",
            "created_at", "updated_at",
        ]


class PhotoUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Photo
        fields = ["category", "sub_category", "memo"]