# photos/models.py
from django.db import models
from django.contrib.auth.models import User
from gallery.models import Photo as GalleryPhoto

# class Photo(models.Model):
#     """사진 정보"""
#     user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='photos')
    
#     # 파일 정보
#     filename = models.CharField(max_length=255)
#     url = models.TextField()
#     thumb_url = models.TextField()
#     file_size = models.BigIntegerField(null=True, blank=True)
    
#     # 해시 (중복 제거용)
#     file_hash = models.CharField(max_length=64, db_index=True)
#     phash = models.CharField(max_length=16, db_index=True)
#     dhash = models.CharField(max_length=16, db_index=True)
#     ahash = models.CharField(max_length=16, null=True, blank=True)
    
#     # 분류 정보
#     category = models.CharField(max_length=50, null=True, blank=True)
#     sub_category = models.CharField(max_length=100, null=True, blank=True)
    
#     # 메타데이터
#     SOURCE_CHOICES = [
#         ('UPLOAD', 'Upload'),
#         ('GOOGLE', 'Google'),
#     ]
#     source = models.CharField(max_length=20, choices=SOURCE_CHOICES)
#     google_id = models.CharField(max_length=255, null=True, blank=True, unique=True)
    
#     memo = models.TextField(null=True, blank=True)
#     width = models.IntegerField(null=True, blank=True)
#     height = models.IntegerField(null=True, blank=True)
#     taken_at = models.DateTimeField(null=True, blank=True)
    
#     # Exact duplicate 관계 저장
#     duplicate_of = models.ForeignKey(
#         "self",
#         null=True,
#         blank=True,
#         on_delete=models.SET_NULL,
#         related_name="duplicates",
#         help_text="Exact duplicate인 경우, 원본 Photo",
#     )
    
#     # Soft delete
#     is_deleted = models.BooleanField(default=False)
#     deleted_at = models.DateTimeField(null=True, blank=True)
    
#     created_at = models.DateTimeField(auto_now_add=True)
#     updated_at = models.DateTimeField(auto_now=True)
    
#     class Meta:
#         db_table = 'photos'
#         ordering = ['-created_at']
#         unique_together = [['user', 'file_hash']]
#         indexes = [
#             models.Index(fields=['user', 'is_deleted']),
#             models.Index(fields=['category']),
#         ]
    
#     def __str__(self):
#         return f"{self.filename} ({self.user.username})"


class GoogleCredential(models.Model):
    """구글 OAuth 토큰"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='google_credential')
    
    google_email = models.EmailField()
    access_token = models.TextField()
    refresh_token = models.TextField()
    token_uri = models.CharField(max_length=255)
    client_id = models.CharField(max_length=255)
    client_secret = models.CharField(max_length=255)
    scopes = models.JSONField()
    
    is_active = models.BooleanField(default=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'google_credentials'
    
    def __str__(self):
        return f"{self.user.username} - {self.google_email}"


class ImportJob(models.Model):
    """Import 작업 상태"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='import_jobs')
    
    job_id = models.CharField(max_length=100, unique=True)
    
    STATUS_CHOICES = [
        ('QUEUED', 'Queued'),
        ('RUNNING', 'Running'),
        ('DONE', 'Done'),
        ('FAILED', 'Failed'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='QUEUED')
    source = models.CharField(max_length=20, default='GOOGLE')
    folder_id = models.CharField(max_length=255, null=True, blank=True)
    
    total_count = models.IntegerField(default=0)
    done_count = models.IntegerField(default=0)
    skipped_count = models.IntegerField(default=0)
    failed_count = models.IntegerField(default=0)
    
    error_message = models.TextField(null=True, blank=True)
    
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'import_jobs'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'status']),
        ]
    
    def __str__(self):
        return f"Job {self.job_id} - {self.status}"