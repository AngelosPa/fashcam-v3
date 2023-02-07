import streamlit as st
from PIL import Image
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from keras.models import Model
from keras.models import load_model
import cv2
from keras.models import load_model
# import feature_extraction_cosine
import feature_extraction_cosine
from streamlit_cropper import st_cropper
from io import StringIO, BytesIO
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

# function to extract the image once we get it and pass the extracted version to the model


def image_extractor(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = image[np.newaxis, ...]
    img_reshape = img_reshape[..., np.newaxis]

    return img_reshape


st.set_page_config(page_title="Image Recommendation System", layout="wide")

model = load_model("final_model_v3.h5")

image = Image.open('./Fashion_Camera3.jpg')
new_image = image.resize((900, 180))
st.image(new_image)
st.markdown('<h1 style="color: red;font-size: 70px;">FashCam</h1>',
            unsafe_allow_html=True)

st.markdown(
    "A new search engine, developed with a cutting-edge image recognition technology that makes it easy to find the fashion you want. With **:red[_FashCam_]**:camera:, you can simply take picture or upload an image and our algorithm will match it with similar products available for purchase online. It's that simple!")
st.sidebar.write("## Upload or Take a Picture")

choice = st.sidebar.radio(
    "Choose an option", ('Upload an image', 'Take a picture'))
if choice == 'Upload an image':
    file = st.sidebar.file_uploader(
        "Choose an image from your computer", type=["jpg", "jpeg", "png"])

if file:
    st.write(
        "Thank you for uploading the image. Please select any one product at a time to have a look!")
    image = Image.open(file)

    # image = image.resize((250, 250))
    cropped_img = st_cropper(image)
    extracted_img = image_extractor(image)
    st.write("Preview")
    b = BytesIO()
    cropped_img.save(b, format="jpeg")
    final_img = Image.open(b)
    # displaying image

    st.image(final_img, width=150)
    predictions = model.predict(extracted_img)

    softmax = np.exp(predictions) / \
        np.sum(np.exp(predictions), axis=-1, keepdims=True)

    propability = round(softmax[0][0], 2)
    st.write("we are searching in our shop for similar " +
             unique_types[np.argmax(predictions)] + propability)

    similar_pictures = feature_extraction_cosine.get_closest_images(
        extracted_img, 'related-products')

    st.subheader("we are searching in our shop for similar " +
                 'related-products')

  #    display the 3 images in a row
    col1, col2, col3 = st.columns(3)
    if len(similar_pictures):
        closest_img1, closest_img2, closest_img3 = similar_pictures
        with col1:
            st.image(closest_img1)
        with col2:
            st.image(closest_img2)
        with col3:
            st.image(closest_img3)
    else:
        st.write("Sorry, we could not find any similar products")
else:
    st.write("Upload an image")


if choice == 'Take a picture':
    # st.write('Live')
    picture = st.sidebar.camera_input("Take a picture")
    if picture:
        # st.image(picture)
        st.write(
            "Thank you for click the picture. Please select any one product at a time, you want to see!")
        image = Image.open(picture)
        # image = image.resize((250, 250))
        cropped_img = st_cropper(image)
        st.write("Preview")
        b = BytesIO()
        cropped_img.save(b, format="jpeg")
        final_img = Image.open(b)
        # displaying image
        st.image(final_img, width=150)
        extracted_img = image_extractor(file)
        predictions = model.predict(extracted_img)
    similar_pictures = feature_extraction_cosine.get_closest_images(
        extracted_img, 'related-products')

    st.subheader("we are searching in our shop for similar " +
                 'related-products')

  #    display the 3 images in a row
    col1, col2, col3 = st.columns(3)
    if len(similar_pictures):
        closest_img1, closest_img2, closest_img3 = similar_pictures
        with col1:
            st.image(closest_img1)
        with col2:
            st.image(closest_img2)
        with col3:
            st.image(closest_img3)
    else:
        st.write("Sorry, we could not find any similar products")
else:
    st.write("Take a picture")
