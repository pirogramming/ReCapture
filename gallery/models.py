from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import timedelta

class Category(models.Model):
    # 6가지 고정 대분류 정의
    BASIC_CATEGORIES = [
        ('finance', '결제/예약'),
        ('study_note', '학습/노트'),
        ('info', '정보'),
        ('others', '기타'),
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
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='gallery_photos') # 중복 방지를 위해 이름 변경
    
    # [파일 및 이미지 정보]
    image = models.ImageField(upload_to='photos/%Y/%m/%d/')
    filename = models.CharField(max_length=255)
    url = models.TextField(null=True, blank=True)
    file_size = models.BigIntegerField(null=True, blank=True)
    
    # [해시 데이터 - photos 모델에서 가져옴]
    file_hash = models.CharField(max_length=64, db_index=True, null=True, blank=True)
    phash = models.CharField(max_length=16, db_index=True, null=True, blank=True)
    dhash = models.CharField(max_length=16, db_index=True, null=True, blank=True)
    ahash = models.CharField(max_length=16, null=True, blank=True)
    
    # [분류 및 서비스 정보]
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True, related_name='photos')
    memo = models.TextField(blank=True, null=True)
    is_bookmarked = models.BooleanField(default=False)
    is_confirmed = models.BooleanField(default=False)
    
    # [상태 정보 (휴지통 등)]
    is_trashed = models.BooleanField(default=False)
    trashed_at = models.DateTimeField(blank=True, null=True)
    is_deleted = models.BooleanField(default=False) # Soft delete (photos 모델 호환)
    
    # [소스 정보]
    SOURCE_CHOICES = [('UPLOAD', 'Upload'), ('GOOGLE', 'Google')]
    source = models.CharField(max_length=20, choices=SOURCE_CHOICES, default='UPLOAD')
    google_id = models.CharField(max_length=255, null=True, blank=True, unique=True)
    
    # [메타데이터]
    width = models.IntegerField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)
    taken_at = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'combined_photos' # 테이블 이름 고정
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.filename} ({self.user.username})"

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

class UserSetting(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    
    # 알림 설정 관련
    is_reminder_enabled = models.BooleanField(default=True) # 알림 여부
    reminder_categories = models.ManyToManyField('Category', blank=True) # 선택된 카테고리
    reminder_days = models.IntegerField(default=7) # n일 미확인 시
    
    # 자동 휴지통 설정 관련
    is_auto_trash_enabled = models.BooleanField(default=False) # 이용 여부
    auto_trash_categories = models.ManyToManyField('Category', related_name='trash_settings', blank=True)
    auto_trash_days = models.IntegerField(default=30) # n일 미확인 시 이동
    trash_expiry_days = models.IntegerField(default=30) # 휴지통 비우기 빈도 (3, 15, 30, 0=안함)

    def __str__(self):
        return f"{self.user.username}의 설정"