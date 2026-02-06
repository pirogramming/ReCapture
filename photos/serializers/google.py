# photos/serializers/google.py
from rest_framework import serializers

class GoogleCallbackSerializer(serializers.Serializer):
    """구글 OAuth 콜백 요청"""
    code = serializers.CharField(required=True)
    redirectUri = serializers.CharField(required=True)

class GoogleStatusResponseSerializer(serializers.Serializer):
    """구글 연동 상태 응답"""
    connected = serializers.BooleanField()
    googleEmail = serializers.EmailField(required=False)

class GoogleAuthorizeResponseSerializer(serializers.Serializer):
    """구글 연동 URL 응답"""
    authUrl = serializers.CharField()

class GoogleCallbackResponseSerializer(serializers.Serializer):
    """구글 콜백 완료 응답"""
    connected = serializers.BooleanField()
    googleEmail = serializers.EmailField()

class GoogleUnlinkResponseSerializer(serializers.Serializer):
    """구글 연동 해제 응답"""
    unlinked = serializers.BooleanField()