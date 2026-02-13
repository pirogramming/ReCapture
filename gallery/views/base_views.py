from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from gallery.models import Photo, UserSetting, Category

@login_required
@login_required
def settings_view(request):
    # 설정 객체가 없으면 생성
    setting, created = UserSetting.objects.get_or_create(user=request.user)
    categories = Category.objects.filter(user=request.user)

    if request.method == 'POST':
        # 알림 설정 업데이트
        setting.is_reminder_enabled = request.POST.get('is_reminder') == 'on'
        setting.reminder_days = int(request.POST.get('reminder_days', 7))
        setting.reminder_categories.set(request.POST.getlist('reminder_cats'))

        # 휴지통 설정 업데이트
        setting.is_auto_trash_enabled = request.POST.get('is_auto_trash') == 'on'
        setting.auto_trash_days = int(request.POST.get('auto_trash_days', 30))
        setting.trash_expiry_days = int(request.POST.get('trash_expiry', 30))
        setting.auto_trash_categories.set(request.POST.getlist('trash_cats'))
        
        setting.save()
        return redirect('gallery:settings')

    return render(request, 'gallery/settings.html', {
        'setting': setting,
        'categories': categories
    })

@login_required(login_url='/accounts/login/')
def photo_list(request):
    photos = Photo.objects.filter(user=request.user, is_trashed=False)
    categories = Category.objects.filter(user=request.user, parent=None)

    category_id = request.GET.get('category_id')
    sub_category_id = request.GET.get('sub_category_id')
    is_bookmarked = request.GET.get('bookmarked')

    sub_categories = []
    current_category_name = None

    # 1. 카테고리 필터 적용
    if category_id:
        photos = photos.filter(category_id=category_id)
        sub_categories = Category.objects.filter(user=request.user, parent_id=category_id)

        # 현재 카테고리 이름 가져오기
        try:
            current_cat = Category.objects.get(id=category_id)
            current_category_name = current_cat.name
        except Category.DoesNotExist:
            pass

        if sub_category_id:
            photos = photos.filter(category_id=sub_category_id)

    # 2. 북마크 필터 적용
    if is_bookmarked == 'true':
        photos = photos.filter(is_bookmarked=True)

    categories = Category.objects.filter(user=request.user, parent=None)

    return render(request, 'gallery/photo_list.html', {
        'photos': photos.order_by('-created_at'),
        'categories': categories,
        'sub_categories': sub_categories,
        'current_category': int(category_id) if category_id else None,
        'current_sub_category': int(sub_category_id) if sub_category_id else None,
        'current_category_name': current_category_name,
        'is_bookmarked': is_bookmarked == 'true'
    })

@login_required(login_url='/accounts/login/')
def photo_detail(request, photoid):
    photo = get_object_or_404(Photo, id=photoid, user=request.user)
    return render(request, 'gallery/photo_detail.html', {'photo': photo})