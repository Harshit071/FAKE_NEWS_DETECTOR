# detector_app/urls.py
from django.urls import path
from . import views

app_name = 'detector_app' # Optional: for namespacing if you have many apps

urlpatterns = [
    path('', views.predict_news_view, name='predict_home'),
]