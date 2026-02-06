from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from photos.models import Photo  # photos 앱의 Photo 모델 사용
from django.db import models

@login_required(login_url='/accounts/login/')
def photo_list(request):
    photos = Photo.objects.filter(user=request.user, is_deleted=False)

    category_filter = request.GET.get('category')  # 'unclassified', 'finance', 'study_note', etc.

    # 카테고리 필터 적용
    if category_filter == 'unclassified':
        # 미분류 (category가 null이거나 빈 문자열)
        photos = photos.filter(models.Q(category__isnull=True) | models.Q(category=''))
    elif category_filter:
        # 특정 카테고리
        photos = photos.filter(category=category_filter)

    # 미분류 개수 계산
    unclassified_count = Photo.objects.filter(
        user=request.user,
        is_deleted=False
    ).filter(
        models.Q(category__isnull=True) | models.Q(category='')
    ).count()

    # 카테고리별 개수
    category_counts = {
        'finance': Photo.objects.filter(user=request.user, is_deleted=False, category='finance').count(),
        'study_note': Photo.objects.filter(user=request.user, is_deleted=False, category='study_note').count(),
        'info': Photo.objects.filter(user=request.user, is_deleted=False, category='info').count(),
        'others': Photo.objects.filter(user=request.user, is_deleted=False, category='others').count(),
    }

    return render(request, 'gallery/photo_list.html', {
        'photos': photos.order_by('-created_at'),
        'current_category': category_filter,
        'unclassified_count': unclassified_count,
        'category_counts': category_counts,
    })

@login_required(login_url='/accounts/login/')
def photo_detail(request, photoid):
    photo = get_object_or_404(Photo, id=photoid, user=request.user)
    return render(request, 'gallery/photo_detail.html', {'photo': photo})