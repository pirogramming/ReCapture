from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from gallery.models import Photo, Category

@login_required(login_url='/accounts/login/')
def photo_list(request):
    photos = Photo.objects.filter(user=request.user, is_trashed=False)
    categories = Category.objects.filter(user=request.user, parent=None)

    category_id = request.GET.get('category_id')
    sub_category_id = request.GET.get('sub_category_id')
    is_bookmarked = request.GET.get('bookmarked')

    sub_categories = []

    # 1. 카테고리 필터 적용
    if category_id:
        photos = photos.filter(category_id=category_id)
        sub_categories = Category.objects.filter(user=request.user, parent_id=category_id)

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
        'is_bookmarked': is_bookmarked == 'true'
    })

@login_required(login_url='/accounts/login/')
def photo_detail(request, photoid):
    photo = get_object_or_404(Photo, id=photoid, user=request.user)
    return render(request, 'gallery/photo_detail.html', {'photo': photo})