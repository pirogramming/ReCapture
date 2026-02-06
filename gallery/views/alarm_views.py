# 리마인드 알림 관련

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from gallery.models import Photo, Category, Notification
import json

@login_required
def reminder_page(request):
    reminders = Notification.objects.filter(user=request.user, notif_type='reminder')
    return render(request, 'gallery/reminder_list.html', {'reminders': reminders})

@csrf_exempt
@login_required
def manage_reminders(request):
    # 1. 알림 목록 조회 (GET)
    if request.method == 'GET':
        # 아직 읽지 않은 리마인드 알림 위주로 가져오기
        reminders = Notification.objects.filter(
            user=request.user, 
            notif_type='reminder'
        ).order_by('-created_at')
        
        items = []
        for r in reminders:
            items.append({
                "reminderId": str(r.id),
                "message": r.message,
                "isRead": r.is_read,
                "createdAt": r.created_at.isoformat()
            })
            
        return JsonResponse({
            "success": True,
            "data": {
                "items": items,
                "total": len(items)
            }
        })

    # 2. 알림 등록 (POST)
    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            # 명세서 기준: photoId, remindAt(알림 예정 시각) 등이 포함될 수 있음
            photo_id = data.get('photoId')
            message = data.get('message', '영수증 확인 리마인드입니다.')
            
            # 알림 객체 생성
            new_reminder = Notification.objects.create(
                user=request.user,
                message=message,
                notif_type='reminder',
                is_read=False
            )
            
            return JsonResponse({
                "success": True,
                "data": {
                    "reminderId": str(new_reminder.id),
                    "status": "scheduled"
                }
            }, status=201)
        except json.JSONDecodeError:
            return JsonResponse({"success": False, "error": "Invalid JSON"}, status=400)

    return JsonResponse({"success": False, "error": "Method not allowed"}, status=405)

@csrf_exempt
@login_required
def edit_reminder(request, reminderId):
    reminder = get_object_or_404(Notification, id=reminderId, user=request.user)

    # 3. 알림 수정 (PUT)
    if request.method == 'PUT':
        data = json.loads(request.body)
        reminder.message = data.get('message', reminder.message)
        reminder.is_read = data.get('isRead', reminder.is_read)
        reminder.save()
        return JsonResponse({"success": True, "data": {"reminderId": reminderId, "updated": True}})

    # 4. 알림 삭제 (DELETE)
    elif request.method == 'DELETE':
        reminder.delete()
        return JsonResponse({"success": True, "data": {"reminderId": reminderId, "deleted": True}})

    return JsonResponse({"success": False, "error": "Method not allowed"}, status=405)