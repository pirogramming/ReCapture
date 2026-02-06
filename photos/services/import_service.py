# photos/services/import_service.py
import uuid
from datetime import datetime

class ImportService:
    """Import 작업 관련 헬퍼 함수"""
    
    @staticmethod
    def generate_job_id():
        """고유한 Job ID 생성"""
        return f"imp_{uuid.uuid4().hex[:12]}"
    
    @staticmethod
    def calculate_progress(job):
        """진행률 계산"""
        if job.total_count == 0:
            return {
                "total": 0,
                "done": 0,
                "skippedDuplicate": 0
            }
        
        return {
            "total": job.total_count,
            "done": job.done_count,
            "skippedDuplicate": job.skipped_count
        }
    
    @staticmethod
    def update_job_status(job, status, error_message=None):
        """Job 상태 업데이트"""
        job.status = status
        
        if status == 'RUNNING' and not job.started_at:
            job.started_at = datetime.now()
        
        if status in ['DONE', 'FAILED']:
            job.completed_at = datetime.now()
        
        if error_message:
            job.error_message = error_message
        
        job.save()
        return job