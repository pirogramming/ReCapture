from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import timedelta

class Category(models.Model):
    # 6가지 고정 대분류 정의
    BASIC_CATEGORIES = [
        ('finance', '결제/금융'),
        ('study_note', '학습/노트'),
        ('shopping', '쇼핑 정보'),
        ('schedule', '일정/예약'),
        ('document', '문서/정보'),
        ('others', '기타정보(비정보)'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='categories')
    name = models.CharField(max_length=50) # 사용자가 보는 이름 (예: 예적금)
    
    # 1차 대분류인 경우에만 선택, 2차 커스텀 폴더일 경우 부모를 따라가거나 비워둠
    category_key = models.CharField(
        max_length=20, 
        choices=BASIC_CATEGORIES,
        null=True, blank=True 
    )

    # 자기 참조: 이 필드가 있으면 2차 분류, 없으면(None) 1차 분류가 됨
    parent = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='subcategories')
    is_bookmarked = models.BooleanField(default=False)

    class Meta:
        # 같은 유저 내에서, 같은 부모 아래에 동일한 이름의 폴더를 만들 수 없도록 제한
        unique_together = ('user', 'name', 'parent')

    def __str__(self):
        return f"[{self.get_category_key_display()}] {self.name}" if self.category_key else self.name

class Photo(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='photos')
    image = models.ImageField(upload_to='photos/%Y/%m/%d/')
    image_hash = models.CharField(max_length=64, unique=True)
    source = models.CharField(max_length=20, choices=[('direct', 'Direct'), ('google', 'Google Drive')])
    
    # 새로 정의한 Category 모델 연결
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True, related_name='photos')
    
    # 기존 문자열 필드는 데이터 마이그레이션 후 삭제해도 무방합니다.
    is_confirmed = models.BooleanField(default=False)
    memo = models.TextField(blank=True, null=True)
    is_bookmarked = models.BooleanField(default=False)
    
    is_trashed = models.BooleanField(default=False)
    trashed_at = models.DateTimeField(blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"[{self.user.username}] {self.id} ({self.category.name if self.category else '미분류'})"

    @property
    def expires_at(self):
        if self.trashed_at:
            return self.trashed_at + timedelta(days=30)
        return None

    def check_auto_trash(self, days=7):
        if not self.is_confirmed and self.created_at <= timezone.now() - timedelta(days=days):
            self.is_trashed = True
            self.trashed_at = timezone.now()
            self.save()
            return True
        return False

class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    photo = models.ForeignKey(Photo, on_delete=models.CASCADE, null=True, blank=True, related_name='reminders')
    message = models.CharField(max_length=255)
    notif_type = models.CharField(max_length=20, choices=[('reminder', '리마인드'), ('trash', '휴지통이동')])
    remind_at = models.DateTimeField(null=True, blank=True)
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"[{self.user.username}] {self.message}"