from django.urls import path

from GunDetection import views

urlpatterns = [
    path('', views.index, name='index'),
    path('start', views.start, name='videoStart'),
    path('stop', views.stop, name='videoStop'),

]
