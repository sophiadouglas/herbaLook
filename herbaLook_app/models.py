from django.db import models

# Create your models here.
class herbariumFolders(models.Model):
    name = models.CharField(max_length=100)
    species = models.CharField(max_length=100)
    genus = models.CharField(max_length=100)
    family = models.CharField(max_length=100)
    

class herbariumImages(models.Model):
    folder = models.ForeignKey(herbariumFolders, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='images')

