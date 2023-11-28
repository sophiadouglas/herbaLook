![herbaLook](https://github.com/sophiadouglas/herbaLook/blob/a46973362538465bfef1440c732c01c82b78765c/static/images/herbaLook.png)

# Herbarium-Field Cross-Domain Plant Classification Web App
*<h4 align="center">herbaLook: a look at herbaria</h4>*

## Description
herbaLook is part of a cross-domain plant identification research involving dried plant specimens (herbarium images) to identify plant photographs in the field with triplet networks.

The proposed triplet network, namely, the Herbarium-Field Triplet Loss (HFTL) Network, is constructed with two Convolutional Neural Networks that map the herbarium and field domains together.

The aim of herbaLook is to demonstrate the trained HFTL model as well as its practicality for taxonomists and fellow naturalists. 

In addition, it is also part of the fulfilment of the requirements 
for my Master Research Degree. More info about the work can be found [here](https://doi.org/10.1007/s00521-022-07951-6).
<p align="center">
<img src=https://github.com/sophiadouglas/herbaLook/blob/a46973362538465bfef1440c732c01c82b78765c/static/images/HFTL_network.png height="300">
</p>

## Website
[https://herbalook.neuon-kuching.com/](https://herbalook.neuon-kuching.com/)

## Usage
- Upload a plant image (field domain) and get up to Top-50 prediction results from 997 plant species
- View the similarity scores between the uploaded image and the herbarium dictionary (dataset/herbaria collection)
  *note that classifications by HFTL models are based on similarity scores instead of probabilities*
- Analyze the HFTL model's prediction through the Class Activation Map visualizations
- View a sample of the dataset images
- View the list of trained species
- View species sharing the same Genus and Family (identify visually similar species that may be commonly misclassified)

## Acknowledgements
This research is made possible with the support from Swinburne University of Technology Sarawak Campus and [NEUON AI Sdn. Bhd., Malaysia](https://github.com/NeuonAI).

This application uses the dataset (herbarium and field images) from [PlantCLEF 2021](https://www.imageclef.org/PlantCLEF2021) and can be acquired at [AIcrowd](https://www.aicrowd.com/challenges/lifeclef-2021-plant).

Cross-domain plant classification remains [a challenging task](https://ceur-ws.org/Vol-2936/paper-122.pdf) and still requires further research as its prediction accuracies are significantly lower than 
conventional automated plant classification. Nevertheless, the HFTL approach offers a step in alleviating the challenging task of automated 
plant identification with few field image samples, specifically rare species, which require high-level expertise. More research is welcomed and anticipated for the plant biodiversity community in the times ahead.


## herbaLook Installation (Local)
Windows
1. [Install Python 3.7.16](https://www.digitalocean.com/community/tutorials/install-python-windows-10)
2. [Set up a virtual environment](https://realpython.com/python-virtual-environments-a-primer/)
3. Activate the virtual environment
4. Install the dependencies
```
pip install requirements.txt
```
5. Download this repository
6. Download the prediction model + dictionary (model.zip), and database images (images.zip) [here](https://drive.google.com/drive/folders/1cUxrPgfq9XIM67XgZ0FJ1t-7ovVFzCi8?usp=drive_link)
7. Place both zip files in the herbaLook\media directory
8. Unzip each files
10. From the virtual environment, go to the downloaded repository
```
cd path\to\herbaLook\in\your\PC
```
7. Create a new SECRET_KEY by running the following command
```
python manage.py shell -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```
8. Copy and save the printed key to a .env file
```
> Right click on the herbaLook root directory (path\to\herbaLook)
> Create a new Text Document
> Type the following into the Text Document
```
```
SECRET_KEY=the printed secret key
```
```
> Save the Text Document File Name as '.env' and File Type as 'All Files (*)'
```
9. Go back to the virtual environment, make sure you're in the root directory (path\to\herbaLook)
10. Link the database (media\images) to your app
```
python manage.py linkdirectories
```
11. Start the herbaLook web app
```
python manage.py runserver
```
12. View the app by typing the following in your internet browser
```
http://127.0.0.1:8000/
```
