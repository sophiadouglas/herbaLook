from pathlib import Path
from django.contrib import admin
from .models import herbariumFolders, herbariumImages
import os
# Register your models here.


# Define the admin class for the Folder model
@admin.register(herbariumFolders)
class FolderAdmin(admin.ModelAdmin):
    list_display = ('name', 'species', 'genus', 'family',)

# Define the admin class for the Image model
@admin.register(herbariumImages)
class ImageAdmin(admin.ModelAdmin):
    list_display = ('folder', 'image',)
    list_filter = ('folder',)
