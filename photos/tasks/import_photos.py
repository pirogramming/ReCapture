# photos/tasks/import_photos.py
from celery import shared_task
from django.contrib.auth.models import User
from photos.models import GoogleCredential, ImportJob, Photo
from photos.services.google_photos_service import GooglePhotosService
from photos.services.import_service import ImportService
from datetime import datetime
import os

@shared_task(bind=True)
def import_google_photos_task(self, user_id, job_id, folder_id=None, dedupe=True):
    """
    구글 포토에서 사진 가져오기 백그라운드 작업
    
    Args:
        user_id: 사용자 ID
        job_id: Import Job ID
        folder_id: 구글 폴더 ID (선택)
        dedupe: 중복 제거 여부
    """
    
    try:
        # User 및 Job 조회
        user = User.objects.get(id=user_id)
        job = ImportJob.objects.get(job_id=job_id)
        
        # Job 상태를 RUNNING으로 변경
        ImportService.update_job_status(job, 'RUNNING')
        
        # 구글 인증 정보 가져오기
        google_cred = GoogleCredential.objects.get(user=user, is_active=True)
        
        # GooglePhotosService 초기화
        google_service = GooglePhotosService(google_cred)
        
        # 구글 포토에서 사진 목록 가져오기
        page_token = None
        all_items = []
        
        while True:
            result = google_service.get_photos(page_size=100, page_token=page_token)
            items = result.get('items', [])
            all_items.extend(items)
            
            page_token = result.get('nextPageToken')
            if not page_token:
                break
        
        # total_count 업데이트
        job.total_count = len(all_items)
        job.save()
        
        # 사진 다운로드 및 저장
        for idx, media_item in enumerate(all_items):
            try:
                # 사진 정보 추출
                google_id = media_item.get('id')
                filename = media_item.get('filename', f'photo_{google_id}.jpg')
                
                # 중복 확인
                if dedupe:
                    existing = Photo.objects.filter(
                        user=user,
                        google_id=google_id,
                        is_deleted=False
                    ).exists()
                    
                    if existing:
                        job.skipped_count += 1
                        job.save()
                        continue
                
                # 저장 경로 생성
                save_dir = f'storage/photos/{user.id}'
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, filename)
                
                # 사진 다운로드
                google_service.download_photo(media_item, save_path)
                
                # TODO: 해시 계산, 썸네일 생성 등은 박정해님 파트와 연동
                
                # DB에 저장 (임시)
                Photo.objects.create(
                    user=user,
                    filename=filename,
                    url=save_path,
                    thumb_url='',  # TODO: 썸네일 생성 후 추가
                    file_hash='',  # TODO: 해시 계산 후 추가
                    phash='',
                    dhash='',
                    source='GOOGLE',
                    google_id=google_id
                )
                
                job.done_count += 1
                job.save()
                
            except Exception as e:
                job.failed_count += 1
                job.save()
                print(f"Failed to import photo {media_item.get('id')}: {e}")
                continue
        
        # 작업 완료
        ImportService.update_job_status(job, 'DONE')
        
        return {
            'total': job.total_count,
            'done': job.done_count,
            'skipped': job.skipped_count,
            'failed': job.failed_count
        }
    
    except Exception as e:
        # 작업 실패
        job = ImportJob.objects.get(job_id=job_id)
        ImportService.update_job_status(job, 'FAILED', error_message=str(e))
        raise