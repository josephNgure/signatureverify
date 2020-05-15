from django.urls import path
from . import views

app_name = 'project'

urlpatterns = [
    path('', views.project_index, name="project index"),
    # path('/<int:pk>/' ,views.project_detail, name="project detail")

]
