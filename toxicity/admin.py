from django.contrib import admin
from .models import UserRating

@admin.register(UserRating)
class UserRatingAdmin(admin.ModelAdmin):
    list_display = ('email', 'text', 'lime_rating', 'shap_rating', 'deeplift_rating', 'integrated_gradients_rating', 'created_at')
    search_fields = ('email', 'text')  # Поиск по email и тексту
    list_filter = ('created_at',)  # Фильтр по дате