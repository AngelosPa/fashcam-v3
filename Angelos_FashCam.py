# import streamlit as st
# from PIL import Image
# import tensorflow as tf
# from PIL import Image, ImageOps
# import numpy as np
# import webbrowser
# import pandas as pd
# from keras.models import Model
# import os
# from io import StringIO, BytesIO
# from keras.applications.imagenet_utils import preprocess_input
# import cv2
# from keras.models import load_model, model_from_json
# import feature_extraction_cosine
# from streamlit_cropper import st_cropper

# # resnet50
# #st.set_option('deprecation.showfileUploaderEncoding', False)

# df = pd.read_csv('styles.csv', error_bad_lines=False)


# st.set_page_config(page_title="Image Recommendation System", layout='wide')

# # @st.cache(allow_output_mutation=True)
# # model = load_model("final_model.h5")
# json_file = open('model_fashion_angelos.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load weights into new model
# # model.load_weights("final_model_angelos.h5")

# # col1, mid, col2 = st.columns([1, 15, 100])
# # with col1:
# #st.image('./Fashion_Camera2.jpg', width=150)
# image = Image.open('./Pic2.jpg')
# new_image = image.resize((900, 180))
# st.image(new_image)
# # with col2:
# #st.write('A Name')
# st.markdown('<h1 style="color: red;font-size: 60px;">FashCam</h1>',
#             unsafe_allow_html=True)
# #st.markdown('**:black[_...an Image Search Engine_]**.')
# # st.markdown('<h1 style="color: black;font-size: 25px;">...an Image Search Engine</h1>',
# # unsafe_allow_html=True)

# st.markdown(
#     "A new search engine, developed with a cutting-edge image recognition technology that makes it easy to find the fashion you want. With **:red[_FashCam_]**:camera:, you can simply take picture or upload an image and our algorithm will match it with similar products available for purchase online. It's that simple!")
# st.sidebar.write("## Upload an Image or Take a Picture")


# def import_and_predict(image_data, model):
#     size = (224, 224)
#     image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
#     image = np.asarray(image)
#     # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     img_reshape = img[np.newaxis, ...]
#     img_reshape = img_reshape[..., np.newaxis]
#     prediction = model.predict(img_reshape)
#     return prediction


# choice = st.sidebar.radio(
#     "Choose an option", ('Upload an image', 'Take a picture'))
# if choice == 'Upload an image':
#     file = st.sidebar.file_uploader(
#         "Choose an image from your computer", type=["jpg", "jpeg", "png"])
# # if file is None:
# #     #st.sidebar.subheader(
# #         #"Please upload a product image using the browse button :point_up:")
# #     st.sidebar.write(
# #         "Sample images can be found [here](https://github.com/prachiagrl83/WBS/tree/Prachi/Sample_images)!")
# # else:
#     if file:
#         st.write(
#             "Thank you for uploading the image. Please select any one product at a time to have a look!")
#     #st.subheader("Scroll down to see the Top Similar Products...")
#     #st.image(file, width=250)
#         image = Image.open(file)
#         image = image.resize((250, 250))
#         cropped_img = st_cropper(image)
#         st.write("Preview")
#         b = BytesIO()
#         cropped_img.save(b, format="jpeg")
#         final_img = Image.open(b)
#         st.image(final_img, width=150)
#         predictions = import_and_predict(final_img, model)
#         st.subheader("We are searching in our shop for similar " +
#                      unique_types[np.argmax(predictions)])

#         # display the 3 images in a row
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.image(feature_extraction_cosine.get_closest_images(
#                 image, unique_types[np.argmax(predictions)])[0], width=150, use_column_width=True)
#         with col2:
#             st.image(feature_extraction_cosine.get_closest_images(
#                 image, unique_types[np.argmax(predictions)])[1], width=150, use_column_width=True)
#         with col3:
#             st.image(feature_extraction_cosine.get_closest_images(
#                 image, unique_types[np.argmax(predictions)])[2], width=150, use_column_width=True)
#     else:
#         st.write("Upload an image")

# if choice == 'Take a picture':
#     # st.write('Live')
#     picture = st.sidebar.camera_input("Take a picture")
#     if picture:
#         # st.image(picture)
#         st.write(
#             "Thank you for click the picture. Please select any one product at a time, you want to see!")
#         image = Image.open(picture)
#         image = image.resize((250, 250))
#         cropped_img = st_cropper(image)
#         st.write("Preview")
#         b = BytesIO()
#         cropped_img.save(b, format="jpeg")
#         final_img = Image.open(b)
#         st.image(final_img, width=150)
#         predictions = import_and_predict(final_img, model)
#         st.write("we are searching in our shop for similar " +
#                  unique_types[np.argmax(predictions)])
#         # display the 3 images in a row
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.image(feature_extraction_cosine.get_closest_images(
#                 image, unique_types[np.argmax(predictions)])[0], width=150, use_column_width=True)
#         with col2:
#             st.image(feature_extraction_cosine.get_closest_images(
#                 image, unique_types[np.argmax(predictions)])[1], width=150, use_column_width=True)
#         with col3:
#             st.image(feature_extraction_cosine.get_closest_images(
#                 image, unique_types[np.argmax(predictions)])[2], width=150, use_column_width=True)
#     else:
#         st.write("Take a picture")
