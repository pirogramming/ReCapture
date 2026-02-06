from .models import Notification

def notification_context(request):
    if request.user.is_authenticated:
        return {
            'latest_notifications': Notification.objects.filter(user=request.user)[:5],
            'unread_notifications_count': Notification.objects.filter(user=request.user, is_read=False).count()
        }
    return {}