from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from gallery.models import Photo
import json

# 1. 휴지통 목록 조회 (GET)
@csrf_exempt
@login_required
def trash_list(request):
    # GET: 휴지통 목록 화면 보여주기
    if request.method == 'GET':
        trashed_photos = Photo.objects.filter(user=request.user, is_trashed=True)
        return render(request, 'gallery/trash_list.html', {'photos': trashed_photos})

    # POST: 휴지통으로 이동시키기
    elif request.method == 'POST':
        return move_to_trash(request)

# 2. 휴지통 이동 (POST)
@csrf_exempt
@login_required
def move_to_trash(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            photo_id = data.get('photoId')
            photo = get_object_or_404(Photo, id=photo_id, user=request.user)
            photo.is_trashed = True
            photo.trashed_at = timezone.now()
            photo.save()
            return JsonResponse({
                "success": True,
                "data": {"photoId": photo.id, "trashed": True, "expiresAt": photo.expires_at.isoformat()}
            })
        except json.JSONDecodeError:
            return JsonResponse({"success": False, "error": "Invalid JSON"}, status=400)

# 3. 복구 (POST)
@csrf_exempt
@login_required
def restore_photo(request, photoid):
    if request.method == 'POST':
        photo = get_object_or_404(Photo, id=photoid, user=request.user, is_trashed=True)
        photo.is_trashed = False
        photo.trashed_at = None
        photo.save()
        return JsonResponse({"success": True, "data": {"photoId": photo.id, "restored": True}})

# 4. 영구 삭제 (DELETE)
@csrf_exempt
@login_required
def permanent_delete(request, photoid):
    if request.method == 'DELETE':
        photo = get_object_or_404(Photo, id=photoid, user=request.user, is_trashed=True)
        if photo.image:
            photo.image.delete()
        photo.delete()
        return JsonResponse({"success": True, "data": {"photoId": photoid, "deletedPermanently": True}})