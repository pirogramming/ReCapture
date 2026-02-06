from django.core.management.base import BaseCommand
from gallery.models import Photo, Notification

class Command(BaseCommand):
    help = '7일 이상 미확인된 사진을 휴지통으로 이동합니다.'

    def handle(self, *args, **options):
        # 미확인 상태의 사진들만 가져오기
        photos = Photo.objects.filter(is_confirmed=False, is_trashed=False)
        count = 0
        for photo in photos:
            if photo.check_auto_trash(days=0): # 7일 기준
                count += 1
                Notification.objects.create(
                    user=photo.user,
                    message=f"미확인 사진이 휴지통으로 이동되었습니다: {photo.created_at.strftime('%Y-%m-%d')}"
                )
        
        self.stdout.write(self.style.SUCCESS(f'총 {count}장의 사진이 휴지통으로 이동되었습니다.'))