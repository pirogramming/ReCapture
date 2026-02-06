# photos/serializers/response.py
from rest_framework import serializers
from typing import TypeVar, Generic, Optional

T = TypeVar('T')

class APIResponse:
    """공통 응답 포맷"""
    
    @staticmethod
    def success(data):
        """성공 응답"""
        return {
            "success": True,
            "data": data
        }
    
    @staticmethod
    def error(code: str, message: str):
        """실패 응답"""
        return {
            "success": False,
            "error": {
                "code": code,
                "message": message
            }
        }