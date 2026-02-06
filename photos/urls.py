# photos/urls.py
from django.urls import path
from photos.api import google, upload, import_job, photos, dedupe, classify

app_name = 'photos'

urlpatterns = [
    # 구글 연동 (1~4번)
    path('google/status', google.google_status, name='google-status'),
    path('google/authorize', google.google_authorize, name='google-authorize'),
    path('google/callback', google.google_callback, name='google-callback'),
    path('google/unlink', google.google_unlink, name='google-unlink'),
    
    # 직접 업로드 (5번)
    path('upload', upload.upload_photos, name='upload-photos'),
    
    # Import Job (6~7번)
    path('import/google', import_job.import_from_google, name='import-google'),
    path('import/jobs/<str:job_id>', import_job.get_import_job_status, name='import-job-status'),
    
    # 사진 CRUD (8~10, 12번)
    path('', photos.list_photos, name='list-photos'),
    path('<int:photo_id>', photos.get_photo_detail, name='photo-detail'),
    path('<int:photo_id>/update', photos.update_photo, name='update-photo'),
    path('<int:photo_id>/delete', photos.delete_photo, name='delete-photo'),
    
    # 중복 제거 (11번)
    path('dedupe/check', dedupe.check_duplicates, name='dedupe-check'),
    
    # 자동 분류 (신규)
    path('classify', classify.classify_photos, name='classify-photos'),
    path('classify/count', classify.get_unclassified_count, name='unclassified-count'),
]