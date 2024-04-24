import streamlit as st
import cv2
import numpy as np
import zipfile
import os

# Fonction pour calculer la distance euclidienne entre deux vecteurs
def euclidean_distance(descriptor1, descriptor2):
    return np.linalg.norm(descriptor1 - descriptor2)

# Titre de l'application
st.title("Identification d'individus par l'iris")

# Chargement de l'image requête
st.subheader("Chargement de l'image requête")
image_requete_path = st.file_uploader("Veuillez sélectionner l'image requête", type=["jpg", "jpeg", "png"])

image = cv2.imdecode(np.frombuffer(image_requete_path.read(), np.uint8), cv2.IMREAD_COLOR)
# Display original image
st.header("Original Image")
st.image(image, channels="BGR")       

if image_requete_path:
    # Chargez l'image requête en niveaux de gris
    #image_requete = cv2.imdecode(np.fromstring(image_requete_path.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    
    st.header("Grayscale Image")

    image_requete = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    st.image(image_requete, caption="Image requête", use_column_width=True)
    

    # Extraction des caractéristiques SIFT de l'image requête
    st.subheader("Extraction des caractéristiques SIFT de l'image requête")
    sift = cv2.SIFT_create()
    keypoints_requete, descripteurs_requete = sift.detectAndCompute(image_requete, None)
    st.write(f"Nombre de keypoints dans l'image requête : {len(keypoints_requete)}")
    st.write(f"Nombre de descripteurs dans l'image requête : {descripteurs_requete.shape[0]}")
    
    
    st.subheader('Image with SIFT Keypoints')

    image_with_keypoints = cv2.drawKeypoints(image_requete, keypoints_requete, None)
    st.image(image_with_keypoints, caption='Image with SIFT Keypoints', use_column_width=True)
    
    
    # Extraction des descripteurs de la base de données à partir du fichier ZIP
    st.subheader("Extraction des descripteurs de la base de données à partir du fichier ZIP")
    with zipfile.ZipFile("descripteurs_base_de_donnees.zip", "r") as zip_ref:
        zip_ref.extractall("descripteurs_base_de_donnees_folder")
    
    # Charger les descripteurs extraits
    database_descriptors = np.load(os.path.join("descripteurs_base_de_donnees_folder", "descripteurs_base_de_donnees.npy"))

    # Seuil pour décider de l'acceptation ou du rejet
    threshold = 8000  # À définir selon vos besoins

    # Calcul des distances euclidiennes entre le descripteur de l'image requête et chaque modèle
    distances = [euclidean_distance(descripteurs_requete, descriptor) for descriptor in database_descriptors]

    # Trouver l'indice du modèle avec la distance minimale
    min_distance_index = np.argmin(distances)
    st.subheader('Iris Identification')

    # Vérifier si la distance minimale est inférieure au seuil donné
    if distances[min_distance_index] < threshold:
        # La correspondance est réussie, la personne est identifiée avec succès
        st.success("La personne a été identifiée avec succès !")
        st.write(f"Indice du modèle correspondant : {min_distance_index}")
    else:
        # La correspondance a échoué, la personne n'est pas identifiée
        st.error("La personne n'a pas été identifiée.")
