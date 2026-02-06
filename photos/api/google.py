# photos/api/google.py
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from photos.models import GoogleCredential
from photos.serializers.google import (
    GoogleCallbackSerializer,
    GoogleStatusResponseSerializer,
    GoogleAuthorizeResponseSerializer,
    GoogleCallbackResponseSerializer,
    GoogleUnlinkResponseSerializer
)
from photos.serializers.response import APIResponse
from photos.services.google_photos_service import GooglePhotosService
from django.shortcuts import redirect
import os

# 1. 구글 연동 상태 조회
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def google_status(request):
    """구글 포토 연동 상태 확인"""
    user = request.user
    
    try:
        google_cred = GoogleCredential.objects.get(user=user, is_active=True)
        
        response_data = APIResponse.success({
            "connected": True,
            "googleEmail": google_cred.google_email
        })
        
        return Response(response_data, status=status.HTTP_200_OK)
    
    except GoogleCredential.DoesNotExist:
        response_data = APIResponse.success({
            "connected": False
        })
        
        return Response(response_data, status=status.HTTP_200_OK)


# 2. 구글 연동 시작 (URL 발급)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def google_authorize(request):
    """구글 OAuth 인증 URL 생성"""
    
    try:
        # 환경 변수에서 redirect URI 가져오기
        redirect_uri = os.getenv('GOOGLE_PHOTOS_REDIRECT_URI', 'http://localhost:8000/api/v1/photos/google/callback')
        
        # 인증 URL 생성
        auth_url, state = GooglePhotosService.get_authorization_url(redirect_uri)
        
        # state를 세션에 저장 (CSRF 방지)
        request.session['google_oauth_state'] = state
        
        return redirect(auth_url)
    
    except Exception as e:
        response_data = APIResponse.error(
            code="GOOGLE_AUTH_ERROR",
            message=f"구글 인증 URL 생성 실패: {str(e)}"
        )
        
        return Response(response_data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# 3. 구글 연동 완료 (콜백)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def google_callback(request):
    code = request.query_params.get('code')
    state = request.query_params.get('state')

    if not code:
        return Response(
            APIResponse.error(code="INVALID_REQUEST", message="구글 인증 코드가 없습니다."),
            status=status.HTTP_400_BAD_REQUEST
        )

    saved_state = request.session.get('google_oauth_state')
    if not saved_state or state != saved_state:
        return Response(
            APIResponse.error(code="INVALID_STATE", message="state 값이 일치하지 않습니다."),
            status=status.HTTP_400_BAD_REQUEST
        )

    request.session.pop('google_oauth_state', None)

    redirect_uri = os.getenv(
        'GOOGLE_PHOTOS_REDIRECT_URI',
        'http://localhost:8000/api/v1/photos/google/callback'
    )

    user = request.user

    try:
        token_data = GooglePhotosService.exchange_code_for_tokens(code, redirect_uri)

        GoogleCredential.objects.update_or_create(
            user=user,
            defaults={
                'google_email': token_data['google_email'],
                'access_token': token_data['access_token'],
                'refresh_token': token_data['refresh_token'],
                'token_uri': token_data['token_uri'],
                'client_id': token_data['client_id'],
                'client_secret': token_data['client_secret'],
                'scopes': token_data['scopes'],
                'is_active': True
            }
        )

        return redirect('/gallery/')

    except Exception as e:
        return Response(
            APIResponse.error(code="GOOGLE_CALLBACK_ERROR", message=str(e)),
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# 4. 구글 연동 해제
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def google_unlink(request):
    """구글 포토 연동 해제"""
    user = request.user
    
    try:
        google_cred = GoogleCredential.objects.get(user=user)
        google_cred.is_active = False
        google_cred.save()
        
        response_data = APIResponse.success({
            "unlinked": True
        })
        
        return Response(response_data, status=status.HTTP_200_OK)
    
    except GoogleCredential.DoesNotExist:
        response_data = APIResponse.error(
            code="NOT_CONNECTED",
            message="연동된 구글 계정이 없습니다."
        )
        
        return Response(response_data, status=status.HTTP_404_NOT_FOUND)
    
    except Exception as e:
        response_data = APIResponse.error(
            code="UNLINK_ERROR",
            message=f"연동 해제 실패: {str(e)}"
        )
        
        return Response(response_data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)