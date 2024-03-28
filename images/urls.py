from django.urls import path
from images import views

urlpatterns = [
    path('images/', views.Images.as_view()),
]