from django.urls import path
from . import views

urlpatterns = [
    path('classify/', views.classify_image, name='classify'),
    path('test/', views.test_page, name='test'),
]