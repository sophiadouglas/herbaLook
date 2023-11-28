from django.urls import path
from . import views

urlpatterns = [
    path('', views.view_index, name='index'),
    path('classify_image/', views.view_classify_image, name='classify_image'),
    path('about/', views.view_about, name='about'),
    path('gallery/', views.view_gallery, name='gallery'),
    path('species/<int:folder_name>/', views.view_species_details, name='species_details'),
    path('dataset_info/', views.view_dataset_info, name='dataset_info'),
]

