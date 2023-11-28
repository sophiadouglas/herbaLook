import os
from django.core.management.base import BaseCommand
from herbaLook_app.models import herbariumFolders, herbariumImages
import pandas as pd
from django.conf import settings


HERBARIUM_MEDIA_DIR = settings.HERBARIUM_MEDIA_DIR
CSV_SPECIES_MAP = settings.CSV_SPECIES_MAP
CSV_MAP = settings.CSV_MAP

# Species map info
SPECIES_DF = pd.read_csv(CSV_SPECIES_MAP, sep=',')
SPECIES_FOLDER = SPECIES_DF['class id'].to_list()

MAP_DF = pd.read_csv(CSV_MAP, sep=';')
MAP_SPECIES = MAP_DF['species'].to_list()
MAP_GENUS = MAP_DF['genus'].to_list()
MAP_FAMILY = MAP_DF['family'].to_list()

# Delete existing data
herbariumImages.objects.all().delete()
herbariumFolders.objects.all().delete()

class Command(BaseCommand):
    help = 'Link directories (media) to model (database)'

    def handle(self, *args, **kwargs):
        link_directory_to_model(HERBARIUM_MEDIA_DIR)

def link_directory_to_model(directory_path):
    # Get a list of all folders in the directory
    folder_names = [folder_name for folder_name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder_name))]

    for folder_name in folder_names:
        # Create an instance of Folder for each folder
        species_name = link_species_name_to_folders(folder_name)
        genus_name = link_genus_name_to_folders(folder_name)
        family_name = link_family_name_to_folders(folder_name)
        folder = herbariumFolders.objects.create(name=folder_name, species=species_name, genus=genus_name, family=family_name)

        # Get a list of image files in the folder
        folder_path = os.path.join(directory_path, folder_name)
        image_files = [file_name for file_name in os.listdir(folder_path) if file_name.endswith(('.jpg', '.jpeg', '.png'))]

        # Create an instance of Image for each image and associate it with the folder
        for image_file in image_files:          
            image_path = os.path.join('images', 'herbarium', folder_name, image_file)
            herbariumImages.objects.create(folder=folder, image=image_path)  

def link_species_name_to_folders(folder_name):
    folder_index = SPECIES_FOLDER.index(int(folder_name))
    species_name = MAP_SPECIES[folder_index]
    return species_name

def link_genus_name_to_folders(folder_name):
    folder_index = SPECIES_FOLDER.index(int(folder_name))
    genus_name = MAP_GENUS[folder_index]
    return genus_name

def link_family_name_to_folders(folder_name):
    folder_index = SPECIES_FOLDER.index(int(folder_name))
    family_name = MAP_FAMILY[folder_index]
    return family_name        