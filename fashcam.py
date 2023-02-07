import streamlit as st
from PIL import Image
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from keras.models import Model
from keras.models import load_model
import cv2
from keras.models import load_model, model_from_json
# import feature_extraction_cosine
import feature_extraction_cosine

# function to extract the image once we get it and pass the extracted version to the model


def image_extractor(image):
    image = Image.open(image)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = image[np.newaxis, ...]
    img_reshape = img_reshape[..., np.newaxis]

    return img_reshape


unique_types = ['Backpacks',
                'Belts',
                'Bra',
                'Caps-hats',
                'CasualShoes',
                'Dresses',
                'Earrings',
                'Handbags',
                'Heels',
                'Leggings',
                'Outwear',
                'pijamas',
                'Ring',
                'Sandals',
                'Scarves',
                'Shirts',
                'Shorts',
                'Skirts',
                'Sportswear',
                'Sunglasses',
                'Sweatshirts',
                'Tops',
                'Trousers',
                'Tshirts']
st.set_page_config(page_title="Image Recommendation System", layout="wide")


model = load_model('final_model_v2.h5')
col1, mid, col2 = st.columns([1, 15, 100])
with col1:
    st.image('./Fashion_Camera2.jpg', width=150)
with col2:
    #st.write('A Name')
    st.markdown('<h1 style="color: red;font-size: 70px;">FashCam</h1>',
                unsafe_allow_html=True)
    st.markdown('<h1 style="color: red;font-size: 30px;">...an Image Search Engine</h1>',
                unsafe_allow_html=True)

st.markdown("Our idea is to build a new search engine **:red[_FashCam_]**:camera:.We have developed a cutting-edge image recognition technology that makes it easy to find the fashion you want. With **:red[_FashCam_]**:camera:, you can simply take a picture of an item or upload an image and our algorithm will match it with similar products available for purchase online. It's that simple!")
st.sidebar.write("## Upload or Take a Picture")

# Upload the image file
file = st.sidebar.file_uploader(
    "Choose an image from your computer", type=["jpg", "jpeg", "png"])


def import_and_predict(image_data, model):

    prediction = model.predict(image_data)

    return prediction


if file is None:
    st.sidebar.subheader(
        "Please upload a product image using the browse button :point_up:")
    st.sidebar.write(
        "Sample image can be found [here](https://github.com/prachiagrl83/WBS/tree/Prachi/Sample_images)!")

else:
    # numerate it with the help of the function
    extracted_img = image_extractor(file)
    st.sidebar.subheader(
        "Thank you for uploading the image. Below you can see image which you have just uploaded!")
    st.subheader("Scroll down to see the Top Similar Products...")
    st.sidebar.image(file, width=250)

    # predictions = import_and_predict(extracted_img, model)
    predictions = model.predict(extracted_img)

    result = st.write("we are searching in our shop for similar " +
                      unique_types[np.argmax(predictions)])

# display the 3 images in a row
    col1, col2, col3 = st.columns(3)
    closest_img1, closest_img2, closest_img3 = feature_extraction_cosine.get_closest_images(
        extracted_img, unique_types[np.argmax(predictions)])
    with col1:
        st.image(closest_img1)
    with col2:
        st.image(closest_img2)
    with col3:
        st.image(closest_img3)
