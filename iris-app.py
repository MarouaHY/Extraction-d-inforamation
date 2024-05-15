import streamlit as st
import cv2
import numpy as np
import zipfile
import os

# Function for calculating the Euclidean distance between two vectors
def euclidean_distance(descriptor1, descriptor2):
    return np.linalg.norm(descriptor1 - descriptor2)

# Titre de l'application
st.write('<h1 style="font-family:Arial; color:grey; text-align:center;">Iris Identification System</h1>', unsafe_allow_html=True)
st.write('<h5 style="font-family:Arial; color:grey; text-align:center;">Identify individuals based on their iris patterns.</h5>', unsafe_allow_html=True)

# Chargement de l'image requête
st.subheader("**Upload an Iris Image**"  )
image_requete_path = st.file_uploader("  ", type=["jpg", "jpeg", "png"])

if image_requete_path is not None:  # Check if an image is uploaded
    image = cv2.imdecode(np.frombuffer(image_requete_path.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Display original image
    st.subheader("Original Image")
    st.image(image, channels="BGR")
    
    # Processing details
    if image is not None:  # Check if image is processed
        st.write("Image processing details...")  # Display processing info
    else:
        st.info("Waiting for the processing.")  # Informative message while processing
    
    # Rest of your code for processing the uploaded image...

else:
    st.info(" Please upload an image.")  # Inform user to upload an image


if image_requete_path:
    # Chargez l'image requête en niveaux de gris
    #image_requete = cv2.imdecode(np.fromstring(image_requete_path.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    
    st.subheader("Grayscale Image")

    image_requete = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    st.image(image_requete, caption="Grayscale Image of the Iris", use_column_width=True)
    

    # Extraction des caractéristiques SIFT de l'image requête
    st.subheader("SIFT Keypoint Detection")
    sift = cv2.SIFT_create()
    keypoints_requete, descripteurs_requete = sift.detectAndCompute(image_requete, None)
    st.write(f"Number of Keypoints Detected :  {len(keypoints_requete)}")
    st.write(f"Number of Descriptors Extracted :  {descripteurs_requete.shape[0]}")
    
    
    st.subheader('Image with SIFT Keypoints')

    image_with_keypoints = cv2.drawKeypoints(image_requete, keypoints_requete, None)
    st.image(image_with_keypoints, caption='Image Highlighting Keypoints', use_column_width=True)
    
    
    # Extraction des descripteurs de la base de données à partir du fichier ZIP
    with st.spinner("Comparing Image Features..."):
      st.subheader("Extracting Descriptors from Database")
      with zipfile.ZipFile("descripteurs_base_de_donnees.zip", "r") as zip_ref:
         zip_ref.extractall("descripteurs_base_de_donnees_folder")
         
    
    # Charger les descripteurs extraits
    database_descriptors = np.load(os.path.join("descripteurs_base_de_donnees_folder", "descripteurs_base_de_donnees.npy"))

    # Seuil pour décider de l'acceptation ou du rejet
    threshold = 10000  # À définir selon vos besoins

    # Calcul des distances euclidiennes entre le descripteur de l'image requête et chaque modèle
    distances = [euclidean_distance(descripteurs_requete, descriptor) for descriptor in database_descriptors]

    # Trouver l'indice du modèle avec la distance minimale
    min_distance_index = np.argmin(distances)
    st.subheader('Identification result')

    # Vérifier si la distance minimale est inférieure au seuil donné
    if distances[min_distance_index] < threshold:
        # La correspondance est réussie, la personne est identifiée avec succès
        st.success("Individual Identified Successfully!")
        with st.container():
                    st.write(f"**Matching Database Model Index**: {min_distance_index}")
    else:
        # La correspondance a échoué, la personne n'est pas identifiée
        st.error("Individual Not Identified in Database.")
