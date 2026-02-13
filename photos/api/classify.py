# photos/api/classify.py

from django.db.models import Q
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from gallery.models import Photo, Category  # ✅ 확정!

# ✅ 너희 ensemble import 경로에 맞춰 수정 (파일 위치가 정확히 이 경로면 그대로)
from classification.services.ensemble_classifier import EnsembleClassifier


LABEL_TO_CATEGORY_NAME = {
    "finance": "결제/예약",
    "study_note": "학습/노트",
    "info": "정보",
    "others": "기타",
}


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def classify_unclassified(request):
    user = request.user
    limit = int(request.data.get("limit", 200))

    # '분류 전' 카테고리 (없으면 생성)
    unclassified_cat, _ = Category.objects.get_or_create(user=user, name="분류 전")

    # ✅ 분류 전(또는 category null) 사진만
    qs = (
        Photo.objects.filter(user=user)
        .filter(Q(category=unclassified_cat) | Q(category__isnull=True))
        .order_by("id")[:limit]
    )

    if qs.count() == 0:
        return Response({"success": True, "data": {"ok": 0, "fail": 0, "results": []}})

    clf = EnsembleClassifier()

    ok, fail = 0, 0
    results = []

    for photo in qs:
        try:
            img_path = photo.image.path

            # ✅ 너희 앙상블은 classify(image_path)로 1장 분류
            out = clf.classify(img_path)
            label = out.get("category")

            if not label:
                raise ValueError("EnsembleClassifier returned empty category")

            category_name = LABEL_TO_CATEGORY_NAME.get(label, "기타")
            target_cat, _ = Category.objects.get_or_create(user=user, name=category_name)

            photo.category = target_cat
            photo.save(update_fields=["category"])

            ok += 1
            results.append(
                {
                    "photo_id": photo.id,
                    "label": label,
                    "category": category_name,
                    "confidence": out.get("confidence"),
                }
            )

        except Exception as e:
            fail += 1
            results.append({"photo_id": photo.id, "error": str(e)})

    return Response({"success": True, "data": {"ok": ok, "fail": fail, "results": results}})