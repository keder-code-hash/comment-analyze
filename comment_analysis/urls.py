from django.urls import path
from . import views


urlpatterns=[
    path('analyse/',views.getCommentAnalyse), 
]