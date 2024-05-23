#importation des données depuis drive
from google.colab import drive
drive.mount('/content/drive')
from google.colab import drive
drive.mount('/content/drive')
import zipfile
import os

# Spécifiez le chemin complet du dossier contenant vos fichiers .zip dans Google Drive
zip_folder_path = "/content/drive/MyDrive/database1"

# Spécifiez le chemin complet du dossier de destination pour extraire les fichiers
extracted_folder_path = "/content/drive/MyDrive/imagesextraites"

# Créez le dossier de destination s'il n'existe pas
os.makedirs(extracted_folder_path, exist_ok=True)

# Parcourez les fichiers du dossier spécifié
for root, dirs, files in os.walk(zip_folder_path):
    for file in files:
        if file.endswith(".zip"):
            zip_file_path = os.path.join(root, file)
            # Ouvrez chaque fichier .zip et extrayez son contenu dans le dossier de destination
            with zipfile.ZipFile(zip_file_path, 'r') as zf:
                zf.extractall(extracted_folder_path)

print("Extraction terminée.")
import cv2
import os

# Spécifiez le chemin complet du dossier contenant vos images dans Google Drive
image_folder_path = "/content/drive/MyDrive/imagesextraites"

# Parcourez les fichiers dans le dossier des images
for root, dirs, files in os.walk(image_folder_path):
    for file in files:
        # Assurez-vous que le fichier est une image (vous pouvez vérifier par extension)
        if file.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(root, file)
            # Lisez l'image avec OpenCV
            image = cv2.imread(image_path)
import os
import cv2

# Chemin vers le dossier contenant les images d'iris
dossier_images = "/content/drive/MyDrive/imagesextraites"

# Chemin vers le dossier pour sauvegarder les images prétraitées
dossier_images_pretraitees = "/content/drive/MyDrive/imagepretraites"
os.makedirs(dossier_images_pretraitees, exist_ok=True)

# Parcourir les fichiers dans le dossier
for nom_fichier in os.listdir(dossier_images):
    # Construire le chemin complet de l'image
    chemin_image = os.path.join(dossier_images, nom_fichier)

    # Charger l'image
    image = cv2.imread(chemin_image)

    # Vérifier si l'image est valide
    if image is None:
        print(f"Impossible de charger l'image : {nom_fichier}")
        continue

    # Prétraitement de l'image (par exemple, conversion en niveaux de gris)
    image_pretraitee = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sauvegarder l'image prétraitée dans le nouveau dossier
    chemin_image_pretraitee = os.path.join(dossier_images_pretraitees, nom_fichier)
    cv2.imwrite(chemin_image_pretraitee, image_pretraitee)

    # Afficher un message pour chaque image prétraitée
    print(f"Image prétraitée sauvegardée : {chemin_image_pretraitee}")
!pip install opencv-python
import cv2
import os

# Chemin vers le dossier contenant les images prétraitées
dossier_images_pretraitees = "/content/drive/MyDrive/imagepretraites"

# Initialiser le détecteur SIFT
sift = cv2.SIFT_create()

# Liste pour stocker les descripteurs SIFT de toutes les images
descripteurs_toutes_images = []

# Itérer sur toutes les images prétraitées dans le dossier
for nom_fichier in os.listdir(dossier_images_pretraitees):
    chemin_image_pretraitee = os.path.join(dossier_images_pretraitees, nom_fichier)
    # Charger l'image prétraitée
    image_pretraitee = cv2.imread(chemin_image_pretraitee, cv2.IMREAD_GRAYSCALE)
    if image_pretraitee is not None:
        # Trouver les points d'intérêt et les descripteurs SIFT
        points_cles, descripteurs = sift.detectAndCompute(image_pretraitee, None)
        # Ajouter les descripteurs à la liste
        if descripteurs is not None:
            descripteurs_toutes_images.append(descripteurs)

# Concaténer tous les descripteurs en une seule matrice
descripteurs_toutes_images = np.concatenate(descripteurs_toutes_images, axis=0)

# Afficher le nombre total de descripteurs extraits
print(f"Nombre total de descripteurs SIFT extraits : {descripteurs_toutes_images.shape[0]}")
import numpy as np

# descripteurs_base_de_donnees est une liste (ou un tableau numpy) contenant les descripteurs SIFT de la base de données
# Assurez-vous que chaque descripteur a la même dimension (par exemple, 128 éléments)
descripteurs_base_de_donnees = descripteurs_toutes_images  # Remplacez [...] par votre liste de descripteurs

# Convertir la liste des descripteurs en un tableau numpy
descripteurs_base_de_donnees_np = np.array(descripteurs_base_de_donnees)

# Enregistrer le tableau numpy dans un fichier .npy
np.save("descripteurs_base_de_donnees.npy", descripteurs_base_de_donnees_np)
# Afficher les descripteurs SIFT extraits
for idx, descripteur in enumerate(descripteurs_toutes_images):
    print(f"Descripteur {idx + 1}: {descripteur}")
