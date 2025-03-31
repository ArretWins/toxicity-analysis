from django.db import models

class UserRating(models.Model):
    email = models.EmailField(blank=True, null=True)  # Email пользователя (необязательное поле)
    text = models.TextField()  # Текст, который анализировался
    lime_rating = models.IntegerField(null=True, blank=True)  # Оценка для LIME
    shap_rating = models.IntegerField(null=True, blank=True)  # Оценка для SHAP
    deeplift_rating = models.IntegerField(null=True, blank=True)  # Оценка для DeepLift
    integrated_gradients_rating = models.IntegerField(null=True, blank=True)  # Оценка для Integrated Gradients
    created_at = models.DateTimeField(auto_now_add=True)  # Дата и время отправки

    def __str__(self):
        return f"Rating by {self.email or 'Anonymous'} on {self.created_at}"