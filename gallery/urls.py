from django.urls import path
from . import views

app_name = 'gallery'

urlpatterns = [
    path('', views.photo_list, name='home'),
    
    # 5. 통합 조회 API (전체 사진 + 갤러리 상태)
    path('photos/', views.photo_list, name='photo_list'),
    path('photos/<int:photoid>/', views.photo_detail, name='photo_detail'),
    
    # 1. 사진 북마크
    path('bookmarks/', views.bookmark_list, name='bookmark_list'), # GET: 목록 조회
    path('bookmarks/add/', views.add_bookmark, name='add_bookmark'),  # POST: 추가
    path('bookmarks/<int:photoid>/', views.delete_bookmark, name='delete_bookmark'), # DELETE: 해제
    
    # 2. 사진 메모 (upsert 방식 적용)
    path('memos/photos/<int:photoid>/', views.manage_memo, name='manage_memo'), # PUT: 생성/수정, GET: 조회, DELETE: 삭제
    
    # 3. 리마인드 알림
    path('reminders/', views.manage_reminders, name='manage_reminders'), # POST, GET
    path('reminders/<str:reminderId>/', views.edit_reminder, name='edit_reminder'), # PUT, DELETE
    
    # 4. 휴지통 기능
    path('trash/', views.trash_list, name='trash_list'), # GET: 목록 조회, POST: 임시 이동
    path('trash/<int:photoid>/restore/', views.restore_photo, name='restore_photo'), # POST: 복구
    path('trash/<int:photoid>/', views.permanent_delete, name='permanent_delete'), # DELETE: 영구 삭제

    # 5. 세부 카테고리 만들기
    path('categories/add/', views.add_category, name='add_category'),
]