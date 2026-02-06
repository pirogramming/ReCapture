from django.contrib import admin
from .models import Photo, Category, Notification

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'category_key', 'parent', 'user']
    list_filter = ['category_key', 'user']

@admin.register(Photo)
class PhotoAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'category', 'is_bookmarked', 'is_trashed', 'created_at']
    list_filter = ['user', 'category', 'is_bookmarked', 'is_trashed']
    search_fields = ['memo']
    def formfield_for_foreignkey(self, db_field, request, **kwargs):
        if db_field.name == "category":
            kwargs["queryset"] = Category.objects.filter(user=request.user)
        return super().formfield_for_foreignkey(db_field, request, **kwargs)

admin.site.register(Notification)