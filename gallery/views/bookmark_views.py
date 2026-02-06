from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from gallery.models import Photo, Category
import json

@login_required
def bookmark_list(request):
    if request.method == 'GET':
        bookmarks = Photo.objects.filter(user=request.user, is_bookmarked=True, is_trashed=False)
        return render(request, 'gallery/bookmark_list.html', {'bookmarks': bookmarks})

# 1. 사진 북마크 추가 (POST /gallery/bookmarks)
@csrf_exempt
@login_required
def add_bookmark(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            photo_id = data.get('photoId')
            # 본인 소유이며 휴지통에 있지 않은 사진 확인
            photo = get_object_or_404(Photo, id=photo_id, user=request.user, is_trashed=False)
            
            photo.is_bookmarked = True
            photo.save()
            
            return JsonResponse({
                "success": True,
                "data": {
                    "bookmarkId": f"bm_{photo.id}",
                    "photoId": photo.id,
                    "createdAt": timezone.now().isoformat()
                }
            })
        except (json.JSONDecodeError, KeyError):
            return JsonResponse({"success": False, "error": "Invalid data format"}, status=400)
    
    return JsonResponse({"success": False, "error": "Method not allowed"}, status=405)

# 2. 사진 북마크 해제 (DELETE /gallery/bookmarks/{photoid})
@csrf_exempt
@login_required
def delete_bookmark(request, photoid):
    if request.method == 'DELETE':
        photo = get_object_or_404(Photo, id=photoid, user=request.user)
        photo.is_bookmarked = False
        photo.save()
        
        return JsonResponse({
            "success": True,
            "data": {
                "photoId": photo.id,
                "deleted": True
            }
        })
    
    return JsonResponse({"success": False, "error": "Method not allowed"}, status=405)

# 3. 북마크된 사진 목록 조회 (GET /gallery/bookmarks)
@login_required
def bookmark_list(request):
    if request.method == 'GET':
        bookmarks = Photo.objects.filter(user=request.user, is_bookmarked=True, is_trashed=False)
        
        items = []
        for p in bookmarks:
            items.append({
                "bookmarkId": f"bm_{p.id}",
                "photoId": p.id,
                "thumbnailUrl": p.image.url if p.image else None,
                "createdAt": p.created_at.isoformat()
            })
        
        return JsonResponse({
            "success": True,
            "data": {
                "items": items,
                "page": 1,
                "pageSize": 30,
                "total": len(items)
            }
        })
    
    return JsonResponse({"success": False, "error": "Method not allowed"}, status=405)