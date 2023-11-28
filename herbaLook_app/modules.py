import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os
import base64
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from .utils import get_current_datetime
from .model_initializer import initialize_model, load_herbarium_dictionary
from django.conf import settings



# Time for print logs
CURRENT_TIME_STAMP = get_current_datetime()

# Import file paths
FROZEN_GRAPH_FILEPATH = settings.FROZEN_GRAPH_FILEPATH
HERBARIUM_DICTIONARY_PKL_FILEPATH = settings.HERBARIUM_DICTIONARY_PKL_FILEPATH
CSV_SPECIES_MAP = settings.CSV_SPECIES_MAP
CSV_MAP = settings.CSV_MAP

# Prediction hyperparameters
INPUT_SIZE = settings.INPUT_SIZE
NUMCLASSES = settings.NUMCLASSES
TOP_N_PREDICTIONS = settings.TOP_N_PREDICTIONS

# Species map info
SPECIES_DF = pd.read_csv(CSV_SPECIES_MAP, sep=',')
SPECIES_LABEL = SPECIES_DF['new label'].to_list()
SPECIES_FOLDER = SPECIES_DF['class id'].to_list()
SPECIES_NAME = SPECIES_DF['species'].to_list()

# Family genus map info
MAP_DF = pd.read_csv(CSV_MAP, sep=';')
MAP_SPECIES = MAP_DF['species'].to_list()
MAP_GENUS = MAP_DF['genus'].to_list()
MAP_FAMILY = MAP_DF['family'].to_list()

# Initialize the model and herbarium dictionary when this module is imported
model_session, field_tensor_input, field_tensor_feat_norm, field_tensor_last_layer = initialize_model(FROZEN_GRAPH_FILEPATH)
herbarium_dictionary = load_herbarium_dictionary(HERBARIUM_DICTIONARY_PKL_FILEPATH)

# Species with field images (in training)
TXT_FIELD_SPECIES = settings.TXT_FIELD_SPECIES
FIELD_DF = pd.read_csv(TXT_FIELD_SPECIES, header=None, sep='\n')
FIELD_LIST = FIELD_DF.squeeze().tolist()


def get_image_size(file):
    image = Image.open(file)
    im_width, im_height = image.size
    print(CURRENT_TIME_STAMP, "Image size acquired")

    return im_width, im_height 

def preprocess_img(file):
    # Outputs a single image which is averaged from 10 crops
    images_cropped = []
    image_ori_duplicated = []
    image_ori = None
    im_width, im_height = get_image_size(file)
    
    try:        
        # Reset the file pointer to the beginning
        file.seek(0)
        # Read the uploaded file directly into an OpenCV-compatible format
        file_stream = file.read()
        nparr = np.frombuffer(file_stream, np.uint8)
        im = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        image_ori = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # Resize image
        im = cv2.resize(im,(INPUT_SIZE[0:2]))

        # Convert to RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # 10 corner crops
        flip_im = cv2.flip(im,1)
        im1 = im[0:260,0:260,:]
        im2 = im[0:260,-260:,:]
        im3 = im[-260:,0:260,:]
        im4 = im[-260:,-260:,:]
        im5 = im[19:279,19:279,:]

        imtemp = [cv2.resize(ims, (INPUT_SIZE[0:2])) for ims in (im1,im2,im3,im4,im5)]
        [images_cropped.append(ims) for ims in imtemp]


        flip_im1 = flip_im[0:260,0:260,:]
        flip_im2 = flip_im[0:260,-260:,:]
        flip_im3 = flip_im[-260:,0:260,:]
        flip_im4 = flip_im[-260:,-260:,:]
        flip_im5 = flip_im[19:279,19:279,:]

        flip_imtemp = [cv2.resize(imf, (INPUT_SIZE[0:2])) for imf in (flip_im1,flip_im2,flip_im3,flip_im4,flip_im5)]
        [images_cropped.append(imf) for imf in flip_imtemp]

        # Normalise im
        images_cropped = np.asarray(images_cropped,dtype=np.float32)/255.0
        im = np.asarray(im,dtype=np.float32)/255.0

        # Add a batch dimension as the model expects batch input
        # im = np.expand_dims(im, 0) 

        # Use tile to replicate the array along the first axis
        image_ori_duplicated = np.tile(im, (10, 1, 1, 1))     

        
    except:
        images_cropped = None
        image_ori_duplicated = None
        image_ori = None
        print(CURRENT_TIME_STAMP, "Exception found in preprocess_img function")

    print(CURRENT_TIME_STAMP, "Image crops acquired")

    return images_cropped, image_ori_duplicated, image_ori, im_width, im_height




def get_field_embedding_and_last_layer(processed_img):
    field_embedding, field_last_layer = model_session.run([field_tensor_feat_norm, field_tensor_last_layer], feed_dict={field_tensor_input: processed_img})
    
    field_embedding = np.mean(field_embedding, axis=0)
    field_last_layer = np.mean(field_last_layer, axis=0)
    print(CURRENT_TIME_STAMP, "Embeddings acquired")

    return field_embedding, field_last_layer
    

def normalise_layer(layer):
    print("layer shape:", layer.shape)
    # layer_squeezed = np.squeeze(layer)
    # print("new layer shape:", layer_squeezed.shape)   
    #   normalise the values
    norm_min_whole = layer.min(keepdims=True)
    norm_max_whole = layer.max(keepdims=True)
    layer_normalised = (layer - norm_min_whole)/(norm_max_whole - norm_min_whole)
    print(CURRENT_TIME_STAMP, "Embeddings normalized")

    return layer_normalised

def get_highest_activated_map(normalised_last_layer):
    total_maps = []
    for n in range(normalised_last_layer.shape[2]):
        feature_emb = normalised_last_layer[:,:,n]
        sum_of_emb = np.sum(feature_emb)
        total_maps.append(sum_of_emb)
    
    # Sort activated maps
    sorted_activated_maps = sorted(((value, index) for index, value in enumerate(total_maps)), reverse=True)
    highest_activated_map = sorted_activated_maps[0][1]
    print(CURRENT_TIME_STAMP, "Highest activation map acquired")

    return highest_activated_map

def get_cam_image(normalised_layer, highest_activated_map, im_width, im_height, image_ori):

    field_highest_activated_cam = normalised_layer[:,:,highest_activated_map]
    field_highest_activated_cam = cv2.resize(field_highest_activated_cam, dsize=(im_width, im_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply the 'jet' colormap
    cmap_img = plt.cm.jet_r(field_highest_activated_cam)
    cmap_img = (cmap_img * 255).astype(np.uint8)
    cmap_img = cmap_img[:, :, :3]

    # Merge original image with CAM
    cam_merged = cmap_img.astype(np.float) * 0.5 + image_ori.astype(np.float) * 0.5

    # Convert to base64 encoding
    _, img_encoded = cv2.imencode('.png', cam_merged)
    target_cam = base64.b64encode(img_encoded).decode('utf-8')
    print(CURRENT_TIME_STAMP, "Target CAM acquired")

    return target_cam

def match_herbarium_dictionary(im_embedding_list, herbarium_emb_list):
    similarity = cosine_similarity(im_embedding_list, herbarium_emb_list)
 
    # 1 - Cosine       
    k_distribution = []
    for sim in similarity:
        new_distribution = []
        for d in sim:
            new_similarity = 1 - d
            new_distribution.append(new_similarity)
        k_distribution.append(new_distribution)
        
    k_distribution = np.array(k_distribution)
        
    # Inverse weighting              
    softmax_list = []
    for d in k_distribution:
        inverse_weighting = (1/np.power(d,5))/np.sum(1/np.power(d,5))
        softmax_list.append(inverse_weighting)
    
    softmax_list = np.array(softmax_list)    
    print(CURRENT_TIME_STAMP, "Dictionary matched")

    return softmax_list


def get_prediction(im_embedding):   
    herbarium_dictionary_embs = []
    for i in range(NUMCLASSES):
        k = i
        emb = herbarium_dictionary[k]
        herbarium_dictionary_embs.append(emb)
        
    herbarium_dictionary_embs = np.asarray(herbarium_dictionary_embs)
    im_embedding = np.reshape(im_embedding, (1,500))
    softmax_output = match_herbarium_dictionary(im_embedding, herbarium_dictionary_embs)
    softmax_output_squeezed = np.squeeze(softmax_output)
    topN_pred = softmax_output_squeezed.argsort()[-TOP_N_PREDICTIONS:][::-1]
    topN_prob = np.sort(softmax_output_squeezed)[-TOP_N_PREDICTIONS:][::-1]
    # prob here means similarity score, not probablity (since the HFTL model is used)

    predicted_species_list = []
    predicted_folder_list = []
    predicted_probabilities_list = []

    for pred, prob in zip(topN_pred, topN_prob):
        prob = round(prob,10)
        pred_index = SPECIES_LABEL.index(pred)
        pred_name = SPECIES_NAME[pred_index]
        pred_folder = SPECIES_FOLDER[pred_index]
        
        predicted_species_list.append(pred_name)
        predicted_folder_list.append(pred_folder)
        predicted_probabilities_list.append(prob)
    print(CURRENT_TIME_STAMP, "Predictions acquired")
    
    return predicted_species_list, predicted_folder_list, predicted_probabilities_list



def get_species_name(folder_name):
    species_index = SPECIES_FOLDER.index(int(folder_name))
    species_name = SPECIES_NAME[species_index]

    return species_name

def get_genus_name(folder_name):
    species_index = SPECIES_FOLDER.index(int(folder_name))
    species_name = SPECIES_NAME[species_index]
    genus_name = MAP_GENUS[species_index]

    return genus_name

def get_family_name(folder_name):
    species_index = SPECIES_FOLDER.index(int(folder_name))
    species_name = SPECIES_NAME[species_index]
    family_name = MAP_FAMILY[species_index]

    return family_name


def get_field_list():
    return FIELD_LIST

