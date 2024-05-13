# Biometric Iris Identification

This is a Streamlit web application for biometric iris identification using the Scale-Invariant Feature Transform (SIFT) algorithm. It allows users to upload an image and compares it with a dataset of pre-registered iris images to identify potential matches.

Features:

-Upload Image: Users can upload an image containing an iris for identification.
-Preprocessing and Feature Extraction: The system preprocesses the image, extracts SIFT features of the iris.
-Comparison with Database: The extracted descriptors are compared with those in the database to find matches.
-Real-Time Display: The user interface displays the processing steps in real-time for better understanding.
-Recognition Result: If a match is found, the interface displays the index of the corresponding iris, along with the recognition decision: "Identified" or "Not identified".

The application is deployed using Streamlit Sharing [You can reach the Iris Recognition application here.](https://extraction-d-inforamation.streamlit.app/)
