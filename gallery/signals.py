from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
from .models import Category

@receiver(post_save, sender=User)
def create_default_categories(sender, instance, created, **kwargs):
    if created:
        # 6대 대분류 정의
        default_cats = [
            ('finance', '결제/금융'),
            ('study_note', '학습/노트'),
            ('shopping', '쇼핑 정보'),
            ('schedule', '일정/예약'),
            ('document', '문서/정보'),
            ('others', '기타정보(비정보)'),
        ]
        for key, name in default_cats:
            Category.objects.create(
                user=instance,
                name=name,
                category_key=key,
                parent=None # 대분류이므로 부모 없음
            )