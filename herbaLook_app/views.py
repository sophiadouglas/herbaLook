from django.shortcuts import render, get_object_or_404, Http404
from django.http import HttpResponse
from django.http import JsonResponse
from . modules import *
from . models import herbariumFolders, herbariumImages
import os
from pathlib import Path
from operator import itemgetter
from django.db.models import Q
import numpy as np
from .utils import get_current_datetime


# Time for print logs
CURRENT_TIME_STAMP = get_current_datetime()



print(CURRENT_TIME_STAMP, "Accessing views.py")
# Create your views here.

def view_index(request):
    print(CURRENT_TIME_STAMP, "Accessing Index page")
    return render(request, "index.html")

def view_about(request):
    print(CURRENT_TIME_STAMP, "Accessing About page")
    return render(request, "about.html")

def view_gallery(request):
    print(CURRENT_TIME_STAMP, "Accessing Gallery page")
    if request.is_ajax():
        search_term = request.GET.get('search_term')
        # Create a Q object that checks for partial matches using icontains
        query = Q(species__icontains=search_term) | Q(genus__icontains=search_term) | Q(family__icontains=search_term)
        results = herbariumFolders.objects.filter(query)

        folder_names = []
        folder_species_list = []
        folder_genus_list = []
        folder_family_list = []
        for folder_name in results:
            folder_species = folder_name.species
            folder_genus = folder_name.genus
            folder_family = folder_name.family

            folder_names.append(folder_name.name)
            folder_species_list.append(folder_species)
            folder_genus_list.append(folder_genus)
            folder_family_list.append(folder_family)


        if len(results) > 0:
            species_names_sorted, folder_names_sorted, genus_names_sorted, famiy_names_sorted = (list(x) for x in zip(*sorted(zip(folder_species_list, folder_names, folder_genus_list, folder_family_list),key=itemgetter(0))))

            species_images_sorted = [] 
            for folder_name in folder_names_sorted:
                image_from_folder = herbariumImages.objects.filter(folder__name=str(folder_name))[:1]
                for path in image_from_folder:
                    image_path = path.image.url
                    species_images_sorted.append(image_path)


            data = {
                'folder_names' : folder_names_sorted,
                'folder_species_list' : species_names_sorted,
                'folder_images_list' : species_images_sorted,
                'folder_genus_list' : genus_names_sorted,
                'folder_family_list' : famiy_names_sorted,
            }
        else:
            data = {
                'folder_names' : "",
                'folder_species_list' : "",
                'folder_images_list' : "",
                'folder_genus_list' : "",
                'folder_family_list' : "",
            }            

        # Convert results to a format suitable for AJAX response (e.g., JSON)
        return JsonResponse(data, safe=False)

    else:   
        folders = herbariumFolders.objects.all()
        folder_names = [folder.name for folder in folders]
        folder_species_list = [folder.species for folder in folders]
        folder_genus_list = [folder.genus for folder in folders]
        folder_family_list = [folder.family for folder in folders]

        species_names_sorted, folder_names_sorted, genus_names_sorted, famiy_names_sorted = (list(x) for x in zip(*sorted(zip(folder_species_list, folder_names, folder_genus_list, folder_family_list),key=itemgetter(0))))

        species_images_sorted = [] 
        for folder_name in folder_names_sorted:
            image_from_folder = herbariumImages.objects.filter(folder__name=str(folder_name))[:1]
            for path in image_from_folder:
                image_path = path.image.url
                species_images_sorted.append(image_path)

        zipped_data = zip(species_names_sorted, folder_names_sorted, species_images_sorted, genus_names_sorted, famiy_names_sorted)

        return render(request, "gallery.html", {'data': zipped_data})


def view_classify_image(request):
    print(CURRENT_TIME_STAMP, "Accessing Classifier")

    if request.method == 'POST' and request.FILES.get('image'):
        print(CURRENT_TIME_STAMP, "Image uploaded")
    	# Get the uploaded image
        uploaded_image = request.FILES['image']

        # Perform image preprocessing
        images_cropped, image_ori_duplicated, image_ori, im_width, im_height = preprocess_img(uploaded_image)

        field_embedding, field_last_layer = get_field_embedding_and_last_layer(image_ori_duplicated)

        field_embedding_10_crops, field_last_layer_10_crops = get_field_embedding_and_last_layer(images_cropped)
  
        normalised_field_last_layer_squeezed = normalise_layer(field_last_layer)

        highest_activated_map = get_highest_activated_map(normalised_field_last_layer_squeezed)

        target_cam = get_cam_image(normalised_field_last_layer_squeezed, highest_activated_map, im_width, im_height, image_ori)

        # Perform image classification
        predicted_species_list, predicted_folder_list, predicted_probabilities_list = get_prediction(field_embedding_10_crops)

        print(CURRENT_TIME_STAMP, "Sending predictions")
        selected_images = []
        # Loop through image folder name
        for folder_name in predicted_folder_list:
            folder_images = []
            # Retrive up to 5 images from the folder        
            images_from_folder = herbariumImages.objects.filter(folder__name=str(folder_name))[:5]

            folder_images = [path.image.url for path in images_from_folder]
            selected_images.append(folder_images)


        data = {
            'image_cam' : target_cam,
            'predicted_folders' : predicted_folder_list,
            'predicted_species' : predicted_species_list,
            'predicted_images' : selected_images,
            'predicted_prob' : predicted_probabilities_list,
        }

        return JsonResponse(data)


def view_species_details(request, folder_name):
    print(CURRENT_TIME_STAMP, "Accessing Species Details page")

    family = herbariumFolders.objects.filter(name=folder_name).values('family')
    genus = herbariumFolders.objects.filter(name=folder_name).values('genus')
    species = herbariumFolders.objects.filter(name=folder_name).values('species')

    requested_family = family[0]['family']
    requested_genus = genus[0]['genus']
    requested_species = species[0]['species']

    images_from_folder = herbariumImages.objects.filter(folder__name=str(folder_name))
    folder_images = [path.image.url for path in images_from_folder]

    # query similar family
    results_family = herbariumFolders.objects.filter(family=requested_family)
    similar_family_folders = []
    similar_family_species = []
    similar_family_genus = []
    similar_family_family = []
    for folder_name in results_family:
        folder_species = folder_name.species
        folder_genus = folder_name.genus
        similar_family_folders.append(folder_name.name)
        similar_family_species.append(folder_species)
        similar_family_genus.append(folder_genus)
        similar_family_family.append(requested_family)
    similar_family_folders_sorted, similar_family_species_sorted, similar_family_genus_sorted, similar_family_family_sorted = (list(x) for x in zip(*sorted(zip(similar_family_folders, similar_family_species, similar_family_genus, similar_family_family),key=itemgetter(0))))
    similar_family_images_sorted = [] 
    for folder_name in similar_family_folders_sorted:
        image_from_folder = herbariumImages.objects.filter(folder__name=str(folder_name))[:1]
        for path in image_from_folder:
            image_path = path.image.url
            similar_family_images_sorted.append(image_path)



    # query similar genus
    results_genus = herbariumFolders.objects.filter(genus=requested_genus)
    similar_genus_folders = []
    similar_genus_species = []
    similar_genus_genus = []
    similar_genus_family = []
    for folder_name in results_genus:
        folder_species = folder_name.species
        folder_family = folder_name.family
        similar_genus_folders.append(folder_name.name)
        similar_genus_species.append(folder_species)
        similar_genus_genus.append(requested_genus)
        similar_genus_family.append(folder_family)
    similar_genus_folders_sorted, similar_genus_species_sorted, similar_genus_genus_sorted, similar_genus_family_sorted = (list(x) for x in zip(*sorted(zip(similar_genus_folders, similar_genus_species, similar_genus_genus, similar_genus_family),key=itemgetter(0))))
    similar_genus_images_sorted = [] 
    for folder_name in similar_genus_folders_sorted:
        image_from_folder = herbariumImages.objects.filter(folder__name=str(folder_name))[:1]
        for path in image_from_folder:
            image_path = path.image.url
            similar_genus_images_sorted.append(image_path)



    data = {
    'family' : requested_family,
    'genus' : requested_genus,
    'species' : requested_species,
    'images' : folder_images,      
    }

    family_data = zip(similar_family_folders_sorted, similar_family_species_sorted, similar_family_genus_sorted, similar_family_family_sorted, similar_family_images_sorted)
    genus_data = zip(similar_genus_folders_sorted, similar_genus_species_sorted, similar_genus_genus_sorted, similar_genus_family_sorted, similar_genus_images_sorted)  
    

    return render(request, 'species_details.html', {'data': data, 'family_data': family_data, 'genus_data': genus_data})



def view_dataset_info(request):
    print(CURRENT_TIME_STAMP, "Accessing Dataset Info page")

    # Get species with herbarium-field training images
    field_list = get_field_list()

    with_field_species = []
    with_field_genus = []
    with_field_family = []

    query = Q(name__in=field_list)
    results = herbariumFolders.objects.filter(query)

    for folder_name in results:
        with_field_species.append(folder_name.species)
        with_field_genus.append(folder_name.genus)
        with_field_family.append(folder_name.family)


    sorted_with_field_species, sorted_with_field_genus, sorted_with_field_family = (list(x) for x in zip(*sorted(zip(with_field_species, with_field_genus, with_field_family),key=itemgetter(0))))


    # Get species without field training images
    without_field_species = []
    without_field_genus = []
    without_field_family = []

    query2 = ~Q(name__in=field_list)
    results2 = herbariumFolders.objects.filter(query2)

    for folder_name in results2:
        without_field_species.append(folder_name.species)
        without_field_genus.append(folder_name.genus)
        without_field_family.append(folder_name.family) 

    sorted_without_field_species, sorted_without_field_genus, sorted_without_field_family = (list(x) for x in zip(*sorted(zip(without_field_species, without_field_genus, without_field_family),key=itemgetter(0))))

    
    # Pack overall data
    data_with_field = zip(sorted_with_field_species, sorted_with_field_genus, sorted_with_field_family)
                       
    data_without_field = zip(sorted_without_field_species, sorted_without_field_genus, sorted_without_field_family)                        

    return render(request, "dataset_info.html", {'data_with_field': data_with_field , 'data_without_field': data_without_field})




   