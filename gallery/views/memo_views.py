# gallery/views/memo_views.py
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from gallery.models import Photo
import json

@csrf_exempt
@login_required
def manage_memo(request, photoid):
    photo = get_object_or_404(Photo, id=photoid, user=request.user)

    # 1. 특정 사진 메모 조회 (GET)
    if request.method == 'GET':
        return JsonResponse({
            "success": True,
            "data": {
                "memo": {
                    "photoId": photo.id,
                    "content": photo.memo,
                    "updatedAt": photo.updated_at.isoformat() if photo.updated_at else None
                }
            }
        })

    # 2. 생성/수정 (PUT)
    elif request.method == 'PUT':
        try:
            data = json.loads(request.body)
            photo.memo = data.get('content')
            photo.save()
            return JsonResponse({
                "success": True, 
                "data": {"photoId": photo.id, "content": photo.memo, "updatedAt": photo.updated_at.isoformat()}
            })
        except json.JSONDecodeError:
            return JsonResponse({"success": False, "error": "Invalid JSON"}, status=400)

    # 3. 삭제 (DELETE)
    elif request.method == 'DELETE':
        photo.memo = None
        photo.save()
        return JsonResponse({"success": True, "data": {"photoId": photo.id, "deleted": True}})

    return JsonResponse({"success": False, "error": "Method not allowed"}, status=405)