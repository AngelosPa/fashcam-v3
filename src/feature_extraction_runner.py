
# TensorFlow and tf.keras


from keras.applications import ResNet50
from tensorflow.keras.utils import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Sequential, model_from_json

from keras.models import Model
from PIL import Image
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from keras.models import Model
# from tensorflow.keras.utils import load_img, img_to_array
# from tensorflow.keras.models import Sequential, model_from_json
import os
from keras.applications.imagenet_utils import preprocess_input
import cv2
from keras.models import load_model, model_from_json
# import resnet50
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
# get the array with the class names from the folder
# read foldernames
import os
import glob
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
unique_types = ['Backpacks',
                'Belts',
                'Bra',
                'Caps-hats',
                'Casual Shoes',
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


def get_prediction_resnet(img_path):

    # Load an image to use for prediction
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Get the predictions from the model
    features = model.predict(x)

    # Print the top 5 predictions
    predictions = decode_predictions(features, top=2)
    # for p in predictions[0]:
    #     print(f"Class: {p[1]}, Probability: {p[2]:.2f}")
    # feature extractor with resnet50
    feat_extractor = Model(
        inputs=model.input, outputs=model.get_layer("avg_pool").output)
    feat_extractor.summary()
    # get the features of the image
    img_features = feat_extractor.predict(x)
    return predictions[0][0], img_features


# function that gets a list of images and returns the features of those images
def get_features(img_paths, category_folder_name):
    importedImages = []
    for f in img_paths:
        filename = f
        original = load_img(filename, target_size=(224, 224))
        numpy_image = img_to_array(original)
        image_batch = np.expand_dims(numpy_image, axis=0)

        importedImages.append(image_batch)
    images = np.vstack(importedImages)
    processed_imgs = preprocess_input(images.copy())
    # load the model

    # Load the pre-trained ResNet50 model, with the top layer removed
    model = ResNet50(weights='imagenet', include_top=True)
    feat_extractor = Model(
        inputs=model.input, outputs=model.get_layer("avg_pool").output)
    feat_extractor.summary()
    imgs_features = feat_extractor.predict(processed_imgs)
    print("features successfully extracted!")
    # save it as csv file
    np.savetxt(f'{category_folder_name}.csv', imgs_features, delimiter=",")
    return imgs_features


# get the image names
# make the path for each image
shopfiles = ['finalDataset/Outwear/' +
             f for f in os.listdir('finalDataset/Outwear')]
get_features(shopfiles, "Outwear")
# create features for every category
# for i in range(len(unique_types)):
#     shopfiles = ['finalDataset/' + unique_types[i] + '/' +
#                  f for f in os.listdir('finalDataset/' + unique_types[i])]
#     get_features(shopfiles, unique_types[i])
