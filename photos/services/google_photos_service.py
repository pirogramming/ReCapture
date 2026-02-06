# photos/services/google_photos_service.py
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
import requests
from datetime import datetime

class GooglePhotosService:
    """Google Photos API 연동 서비스"""
    
    SCOPES = [
        'https://www.googleapis.com/auth/photoslibrary.readonly',
    ]
    
    def __init__(self, google_credential=None):
        """
        Args:
            google_credential: GoogleCredential 모델 인스턴스
        """
        self.google_credential = google_credential
        self.credentials = None
        
        if google_credential:
            self.credentials = self._build_credentials(google_credential)
    
    def _build_credentials(self, google_credential):
        """GoogleCredential 모델로부터 Credentials 객체 생성"""
        return Credentials(
            token=google_credential.access_token,
            refresh_token=google_credential.refresh_token,
            token_uri=google_credential.token_uri,
            client_id=google_credential.client_id,
            client_secret=google_credential.client_secret,
            scopes=google_credential.scopes
        )
    
    @staticmethod
    def get_authorization_url(redirect_uri):
        """OAuth 인증 URL 생성"""
        # ========== 수정: 환경 변수 사용 ==========
        client_config = {
            "web": {
                "client_id": os.getenv('GOOGLE_PHOTOS_CLIENT_ID'),
                "client_secret": os.getenv('GOOGLE_PHOTOS_CLIENT_SECRET'),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [redirect_uri]
            }
        }
        
        flow = Flow.from_client_config(
            client_config,
            scopes=GooglePhotosService.SCOPES,
            redirect_uri=redirect_uri
        )
        # ==========================================
        
        auth_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )
        
        return auth_url, state
    
    @staticmethod
    def exchange_code_for_tokens(code, redirect_uri):
        """인증 코드를 토큰으로 교환"""
        # ========== 수정: 환경 변수 사용 ==========
        client_config = {
            "web": {
                "client_id": os.getenv('GOOGLE_PHOTOS_CLIENT_ID'),
                "client_secret": os.getenv('GOOGLE_PHOTOS_CLIENT_SECRET'),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [redirect_uri]
            }
        }
        
        flow = Flow.from_client_config(
            client_config,
            scopes=GooglePhotosService.SCOPES,
            redirect_uri=redirect_uri
        )
        # ==========================================
        
        flow.fetch_token(code=code)
        credentials = flow.credentials
        
        # 사용자 정보 가져오기
        service = build('oauth2', 'v2', credentials=credentials)
        user_info = service.userinfo().get().execute()
        
        return {
            'access_token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes,
            'google_email': user_info.get('email')
        }
    
    def get_photos(self, page_size=100, page_token=None):
        """Google Photos에서 사진 목록 가져오기"""
        if not self.credentials:
            raise ValueError("Google credentials not set")
        
        try:
            service = build('photoslibrary', 'v1', credentials=self.credentials, static_discovery=False)
            
            results = service.mediaItems().list(
                pageSize=page_size,
                pageToken=page_token
            ).execute()
            
            return {
                'items': results.get('mediaItems', []),
                'nextPageToken': results.get('nextPageToken')
            }
        
        except HttpError as e:
            raise Exception(f"Google Photos API error: {e}")
    
    def download_photo(self, media_item, save_path):
        """사진 다운로드"""
        base_url = media_item.get('baseUrl')
        if not base_url:
            raise ValueError("No baseUrl in media item")
        
        # =d 파라미터로 원본 다운로드
        download_url = f"{base_url}=d"
        
        response = requests.get(download_url)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        return save_path
    
    def get_photos_since(self, since_date, page_size=100):
        """특정 날짜 이후의 사진만 가져오기"""
        if not self.credentials:
            raise ValueError("Google credentials not set")
        
        try:
            service = build('photoslibrary', 'v1', credentials=self.credentials, static_discovery=False)
            
            # 날짜 필터 생성
            filters = {
                'dateFilter': {
                    'ranges': [{
                        'startDate': {
                            'year': since_date.year,
                            'month': since_date.month,
                            'day': since_date.day
                        },
                        'endDate': {
                            'year': datetime.now().year,
                            'month': datetime.now().month,
                            'day': datetime.now().day
                        }
                    }]
                }
            }
            
            results = service.mediaItems().search(
                body={'pageSize': page_size, 'filters': filters}
            ).execute()
            
            return {
                'items': results.get('mediaItems', []),
                'nextPageToken': results.get('nextPageToken')
            }
        
        except HttpError as e:
            raise Exception(f"Google Photos API error: {e}")