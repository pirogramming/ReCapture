from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .services.image_classifier import ImageClassifier
import json

# 분류기 인스턴스 (한번만 로드)
classifier = ImageClassifier()

@csrf_exempt
def classify_image(request):
    """
    이미지 분류 API
    POST /classification/classify/
    """
    if request.method == 'POST':
        try:
            # 이미지 파일 받기
            if 'image' not in request.FILES:
                return JsonResponse({
                    'success': False,
                    'error': '이미지 파일이 없습니다.'
                }, status=400)
            
            image_file = request.FILES['image']
            
            # 분류
            category, category_kr, confidence = classifier.predict_with_korean(image_file)
            
            return JsonResponse({
                'success': True,
                'category': category,
                'category_kr': category_kr,
                'confidence': round(confidence * 100, 2)
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({
        'success': False,
        'error': 'POST 요청만 가능합니다.'
    }, status=405)


def test_page(request):
    """테스트용 HTML 페이지"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>이미지 분류 테스트</title>
        <style>
            body { font-family: Arial; padding: 50px; }
            .container { max-width: 600px; margin: 0 auto; }
            input[type="file"] { margin: 20px 0; }
            button { padding: 10px 20px; font-size: 16px; }
            #result { margin-top: 20px; padding: 20px; background: #f0f0f0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📸 이미지 분류 테스트</h1>
            <input type="file" id="imageInput" accept="image/*">
            <button onclick="classify()">분류하기</button>
            <div id="result"></div>
        </div>
        
        <script>
        async function classify() {
            const input = document.getElementById('imageInput');
            const result = document.getElementById('result');
            
            if (!input.files[0]) {
                alert('이미지를 선택해주세요!');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', input.files[0]);
            
            result.innerHTML = '분류 중...';
            
            try {
                const response = await fetch('/classification/classify/', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    result.innerHTML = `
                        <h3>✅ 분류 결과</h3>
                        <p><strong>카테고리:</strong> ${data.category_kr} (${data.category})</p>
                        <p><strong>확률:</strong> ${data.confidence}%</p>
                    `;
                } else {
                    result.innerHTML = `<p style="color:red;">❌ 오류: ${data.error}</p>`;
                }
            } catch (error) {
                result.innerHTML = `<p style="color:red;">❌ 오류: ${error}</p>`;
            }
        }
        </script>
    </body>
    </html>
    """
    from django.http import HttpResponse
    return HttpResponse(html)