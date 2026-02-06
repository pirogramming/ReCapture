from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from gallery.models import Photo, Category
import json

@csrf_exempt
@login_required
def add_category(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        name = data.get('name')
        parent_id = data.get('parent_id')
        
        parent = get_object_or_404(Category, id=parent_id, user=request.user)

        if Category.objects.filter(user=request.user, parent_id=parent_id, name=name).exists():
            return JsonResponse({"success": False, "error": "이미 존재하는 폴더 이름입니다."}, status=400)
        
        new_cat = Category.objects.create(
            user=request.user,
            name=name,
            parent=parent,
            category_key=f"sub_{timezone.now().timestamp()}" # 임의 키 생성
        )
        return JsonResponse({"success": True, "id": new_cat.id})