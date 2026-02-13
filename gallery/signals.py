from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
from .models import Category

@receiver(post_save, sender=User)
def create_default_categories(sender, instance, created, **kwargs):
    if created:
        # "분류 전" 카테고리 먼저 생성 (최우선)
        Category.objects.create(
            user=instance,
            name='분류 전',
            category_key=None,  # 특수 카테고리로 category_key 없음
            parent=None
        )
        
        # 6대 대분류 정의
        default_cats = [
            ('finance', '결제/금융'),
            ('study_note', '학습/노트'),
            ('info', '문서/정보'),
            ('others', '기타정보(비정보)'),
        ]
        for key, name in default_cats:
            Category.objects.create(
                user=instance,
                name=name,
                category_key=key,
                parent=None
            )