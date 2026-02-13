# gallery/management/commands/auto_process.py

from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from gallery.models import Photo, UserSetting, Notification

class Command(BaseCommand):
    help = '사용자 설정에 따른 자동 리마인드 알림 생성 및 휴지통 이동 처리'

    def handle(self, *args, **options):
        now = timezone.now()
        settings = UserSetting.objects.all()

        for setting in settings:
            user = setting.user
            
            # --- [1] 자동 휴지통 이동 로직 ---
            if setting.is_auto_trash_enabled:
                # 대상 카테고리의 사진 중 아직 휴지통에 가지 않은 사진들
                trash_target_date = now - timedelta(days=setting.auto_trash_days)
                photos_to_trash = Photo.objects.filter(
                    user=user,
                    category__in=setting.auto_trash_categories.all(),
                    is_trashed=False,
                    created_at__lte=trash_target_date
                )
                
                count = photos_to_trash.count()
                for photo in photos_to_trash:
                    photo.is_trashed = True
                    photo.trashed_at = now
                    photo.save()
                
                if count > 0:
                    self.stdout.write(f"{user.username}: {count}장의 사진을 휴지통으로 이동했습니다.")

            # --- [2] 리마인드 알림 생성 로직 ---
            if setting.is_reminder_enabled:
                reminder_target_date = now - timedelta(days=setting.reminder_days)
                # 대상 카테고리 사진 중 아직 확인하지 않은 사진이 있는지 체크
                pending_photos = Photo.objects.filter(
                    user=user,
                    category__in=setting.reminder_categories.all(),
                    is_trashed=False,
                    created_at__lte=reminder_target_date
                ).exists()

                if pending_photos:
                    # 오늘 이미 보낸 리마인드가 있는지 확인 (중복 방지)
                    already_sent = Notification.objects.filter(
                        user=user, 
                        notif_type='reminder', 
                        created_at__date=now.date()
                    ).exists()

                    if not already_sent:
                        Notification.objects.create(
                            user=user,
                            message=f"미확인 영수증이 있습니다. {setting.reminder_days}일이 지났으니 확인해주세요!",
                            notif_type='reminder'
                        )
                        self.stdout.write(f"{user.username}: 리마인드 알림을 생성했습니다.")

            # --- [3] 휴지통 자동 비우기 (영구 삭제) ---
            if setting.trash_expiry_days > 0:
                expiry_date = now - timedelta(days=setting.trash_expiry_days)
                old_trash = Photo.objects.filter(
                    user=user,
                    is_trashed=True,
                    trashed_at__lte=expiry_date
                )
                
                for photo in old_trash:
                    if photo.image:
                        photo.image.delete() # 실제 파일 삭제
                    photo.delete() # DB 레코드 삭제