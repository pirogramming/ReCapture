from django.shortcuts import render, redirect
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.decorators import login_required

@login_required
def mypage(request):
    # 내 정보 조회
    return render(request, 'accounts/mypage.html', {'user': request.user})

def logout_view(request):
    # 로그아웃 처리
    auth_logout(request)
    return redirect('gallery:photo_list')