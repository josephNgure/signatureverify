from django.urls import path
from . import views

urlpatterns=[
    path('home/', views.posts, name='home'),
    path('datey/', views.datey, name='date'),
    path('datey/<int:year>/<str:month>', views.datey),
    path('bootstrap/', views.bootstrap, name='bootstrap'),
]