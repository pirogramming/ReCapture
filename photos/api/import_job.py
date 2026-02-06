# photos/api/import_job.py
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from photos.models import GoogleCredential, ImportJob
from photos.serializers.photo import (
    ImportGoogleRequestSerializer,
    ImportJobStatusResponseSerializer,
    ImportJobStartResponseSerializer
)
from photos.serializers.response import APIResponse
from photos.services.import_service import ImportService
from photos.tasks.import_photos import import_google_photos_task


# 6. 구글 포토에서 사진 가져오기
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def import_from_google(request):
    """구글 포토에서 사진 가져오기 (비동기 작업 시작)"""
    
    serializer = ImportGoogleRequestSerializer(data=request.data)
    
    if not serializer.is_valid():
        response_data = APIResponse.error(
            code="INVALID_REQUEST",
            message="잘못된 요청입니다."
        )
        return Response(response_data, status=status.HTTP_400_BAD_REQUEST)
    
    user = request.user
    folder_id = serializer.validated_data.get('folderId')
    dedupe = serializer.validated_data.get('dedupe', True)
    
    try:
        # 구글 연동 확인
        google_cred = GoogleCredential.objects.get(user=user, is_active=True)
        
    except GoogleCredential.DoesNotExist:
        response_data = APIResponse.error(
            code="GOOGLE_NOT_CONNECTED",
            message="구글 연동이 필요합니다."
        )
        return Response(response_data, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Job ID 생성
        job_id = ImportService.generate_job_id()
        
        # ImportJob 생성
        import_job = ImportJob.objects.create(
            user=user,
            job_id=job_id,
            status='QUEUED',
            source='GOOGLE',
            folder_id=folder_id
        )
        
        # Celery 백그라운드 작업 시작
        import_google_photos_task.delay(
            user_id=user.id,
            job_id=job_id,
            folder_id=folder_id,
            dedupe=dedupe
        )
        
        response_data = APIResponse.success({
            "jobId": job_id,
            "status": "QUEUED"
        })
        
        return Response(response_data, status=status.HTTP_200_OK)
    
    except Exception as e:
        response_data = APIResponse.error(
            code="IMPORT_START_ERROR",
            message=f"Import 작업 시작 실패: {str(e)}"
        )
        return Response(response_data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# 7. Import 작업 진행 상황 조회
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_import_job_status(request, job_id):
    """Import 작업 진행 상황 조회"""
    
    user = request.user
    
    try:
        # 사용자의 Job 조회
        job = ImportJob.objects.get(job_id=job_id, user=user)
        
        # 진행률 계산
        progress = ImportService.calculate_progress(job)
        
        response_data = APIResponse.success({
            "jobId": job.job_id,
            "status": job.status,
            "progress": progress
        })
        
        return Response(response_data, status=status.HTTP_200_OK)
    
    except ImportJob.DoesNotExist:
        response_data = APIResponse.error(
            code="JOB_NOT_FOUND",
            message="작업을 찾을 수 없습니다."
        )
        return Response(response_data, status=status.HTTP_404_NOT_FOUND)
    
    except Exception as e:
        response_data = APIResponse.error(
            code="JOB_STATUS_ERROR",
            message=f"작업 상태 조회 실패: {str(e)}"
        )
        return Response(response_data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)