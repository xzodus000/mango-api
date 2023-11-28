# coding=utf-8
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from flask import Flask, render_template, request
import logging

from sklearn.ensemble import ExtraTreesClassifier

from werkzeug.utils import secure_filename

import cv2
from sklearn import tree
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from math import sqrt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
import time
from collections import Counter
from skimage.feature import hog
from skimage import data, exposure
# from skimage.feature import graycomatrix, graycoprops
import math
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew,kurtosis

from skimage import measure
# import mahotas
import mimetypes
import matplotlib.pyplot as plt

from skimage.filters import gabor
from skimage.color import rgb2gray
from PIL import Image
from keras.callbacks import EarlyStopping
from skimage.feature import graycomatrix, graycoprops

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
file_name = 'static_data_glcm_translucency.csv'



# Get the current working directory
current_dir = os.getcwd()

# Combine the current working directory and the file name to get the full file path
file_path = os.path.join(current_dir, file_name)
print(file_path)



# Get the current working directory
current_dir = os.getcwd()

# List all files in the current directory
files_in_current_dir = os.listdir(current_dir)

# Check if the file is in the current directory
if 'static_data_glcm_translucency.csv' in files_in_current_dir:
    print("File found.")
#     trad_df = pd.read_csv(file_path, index_col = False)
else:
    print("File not found in the current directory.")
    

print(file_path,'toon')
trad_df = pd.read_csv(file_path, index_col = False)
# trad_df.drop("Unnamed: 0",axis=1,inplace=True)?
# trad_df.drop(columns=["label", "filename"], inplace=False)
# trad_df["label"]
trad_df


trad_df = trad_df.drop("path",axis = 1)
trad_df = trad_df.drop("filename",axis = 1)
trad_df

unique_label = np.unique(trad_df['label'])
int_label = []
for l in trad_df['label']:
  result = [x == l for x in unique_label]
  i = np.argmax(result)
  int_label.append(i)

trad_df['label'] = int_label

trad_df.dtypes

trad_df = trad_df.reset_index(drop=True)

trad_df.isnull().sum()

trad_df.replace([np.inf, -np.inf], np.nan, inplace=True)
trad_df.dropna(inplace=True)

# split data for training purpose
from sklearn.model_selection import train_test_split
X = trad_df.drop("label",axis=1)
y = trad_df["label"]
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2)


# model = tf.keras.models.load_model('model')

# model = ExtraTreesClassifier

# modelling

def get_model(index = -1): 
  models = [{
      "name": "SVC",
      "model" : svm.SVC()
  },
  {
      "name" : "RandomForestClassifier",
      "model" : RandomForestClassifier()
  },
  {
      "name" : " KNeighborsClassifier",
      "model" :  KNeighborsClassifier()
  },
  {
      "name" : "DecisionTreeClassifier",
      "model" : tree.DecisionTreeClassifier()
  },
  {
      "name" : "ExtraTreeClassifier",
      "model" : ExtraTreesClassifier()
  },
    {
      "name" : "GradientBoostingClassifier",
      "model" : GradientBoostingClassifier()
  }]

  if index == -1:
    return len(models)

  return models[index]

def train_model(model):
  print("Traning by {} ...".format(model["name"]))

  m = model["model"]
  
  m.fit(X_train,y_train)
  score = m.score(X_val,y_val)
  # Make predictions on the test data
  y_pred = m.predict(X_val)

# Calculate evaluation metrics
  accuracy = accuracy_score(y_val, y_pred)
  precision = precision_score(y_val, y_pred, average='weighted')
  recall = recall_score(y_val, y_pred, average='weighted')
  f1 = f1_score(y_val, y_pred, average='weighted')
#   losses = mean_squared_error(y_val, y_pred,squared=False)
  losses = mean_squared_error(y_val, y_pred)
  loss = f"{losses:.4f}"
  print(f"Accuracy: {accuracy}")
  print(f"Precision: {precision}")
  print(f"Recall: {recall}")
  print(f"F1-score: {f1}")
  print(f"Loss: {loss}")
  print(f"Loss (MSE): {losses:.4f}")

  return m,score,precision,recall,f1,loss

def train_model_single_model(model):
  m = model
  
  m.fit(X_train,y_train)
  score = m.score(X_val,y_val)
  # Make predictions on the test data
  y_pred = m.predict(X_val)

# Calculate evaluation metrics
  accuracy = accuracy_score(y_val, y_pred)
  precision = precision_score(y_val, y_pred, average='weighted')
  recall = recall_score(y_val, y_pred, average='weighted')
  f1 = f1_score(y_val, y_pred, average='weighted')
#   losses = mean_squared_error(y_val, y_pred,squared=False)
  losses = mean_squared_error(y_val, y_pred)
  loss = f"{losses:.4f}"
  print(f"Accuracy: {accuracy}")
  print(f"Precision: {precision}")
  print(f"Recall: {recall}")
  print(f"F1-score: {f1}")
  print(f"Loss: {loss}")
  print(f"Loss (MSE): {losses:.4f}")

  return m,score,precision,recall,f1,loss


myModel = ExtraTreesClassifier()
train_model_single_model(myModel)

## main execution

models = []
scores = []
precisions = []
recalls = []
f1s = []
loss = []
work_model= []

for i in range(get_model()):
  model_wrapper = get_model(i)
  model = model_wrapper["model"]
  model_name = model_wrapper["name"]

  m,score,precision,recall,f1,losses = train_model(model_wrapper)

  models.append(model_name)
  scores.append(score)
  precisions.append(precision)
  recalls.append(recall)
  f1s.append(f1)
  loss.append(losses)
  work_model.append(m)

 



def find_static_data(img,hsv=False):
  #b,g,r
  #h,s,v
  image_resize = cv2.resize(img,(256,256),interpolation = cv2.INTER_AREA)
  mean = image_resize.mean(axis=(0,1))
  sd = image_resize.std(axis=(0,1))
  if hsv:
    h,s,v = cv2.split(img)
    
    flatted_H = h.flatten()
    flatted_S = s.flatten()
    flatted_V = v.flatten()

    s_h = skew(flatted_H)
    s_s = skew(flatted_S)
    s_v = skew(flatted_V)

    k_h = kurtosis(flatted_H)
    k_s = kurtosis(flatted_S)
    k_v = kurtosis(flatted_V)

    payload = {
      'mean' : mean,
      'std' : sd,
      'skew' : [s_h,s_s,s_v],
      'kurtosis' : [k_h,k_s,k_v]
    }
    
  else:
    b,g,r = cv2.split(img)

    flatted_B = b.flatten()
    flatted_G = g.flatten()
    flatted_R = r.flatten()

    s_b = skew(flatted_B)
    s_g = skew(flatted_G)
    s_r = skew(flatted_R)

    k_b = kurtosis(flatted_B)
    k_g = kurtosis(flatted_G)
    k_r = kurtosis(flatted_R)

    payload = {
      'mean' : mean,
      'std' : sd,
      'skew' : [s_b,s_g,s_r],
      'kurtosis' : [k_b,k_g,k_r]
    }

  return payload
  
def calculate_eccentricity(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to obtain a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate the moments of the contour
    moments = cv2.moments(largest_contour)
    
    # Calculate the eccentricity using moments
    major_axis = 2 * ((moments["mu20"] + moments["mu02"] + ((moments["mu20"] - moments["mu02"])**2 + 4 * (moments["mu11"]**2))**0.5) / moments["m00"])**0.5
    minor_axis = 2 * ((moments["mu20"] + moments["mu02"] - ((moments["mu20"] - moments["mu02"])**2 + 4 * (moments["mu11"]**2))**0.5) / moments["m00"])**0.5
    
    # Calculate the eccentricity as the ratio of the major axis to the minor axis
    eccentricity = major_axis / minor_axis
    
    return eccentricity

def compute_haralick_features(image, distances, angles):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the GLCM matrix
    glcm = graycomatrix(gray_image, distances, angles, levels=256, symmetric=True, normed=True)

    # Calculate the Haralick texture features
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        feature = np.mean(graycoprops(glcm, prop))
        features.append(feature)
        
    payload = {
      'Contrast' : features[0],
      'Dissimilarity' : features[1],
      'Homogeneity' : features[2],
      'Energy' : features[3],
      'Correlation' : features[4]
    }
    
    return payload

def extract_contour_features(img):
    # Load the image
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold_value = 128  # Adjust this value based on your image characteristics

    # Apply thresholding to create a binary image
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize a list to store contour-based features
    contour_features_list = []
    
    # Loop through each contour
    for contour in contours:
        # Calculate area of the contour
        area = cv2.contourArea(contour)
        
        # Calculate perimeter of the contour
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate compactness (perimeter^2 / area)
        compactness = (perimeter ** 2) / area if area != 0 else 0.0
        
        # Calculate bounding box dimensions
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio of the bounding box
        aspect_ratio = float(w) / h
        
        # Calculate extent (area of contour / area of bounding box)
        extent = area / (w * h)
        
        # Create a dictionary to store the features
        features_dict = {
            "Area": area,
            "Perimeter": perimeter,
            "Compactness": compactness,
            "Aspect Ratio": aspect_ratio,
            "Extent": extent
        }
        
        # Add the dictionary to the list
        contour_features_list.append(features_dict)
    
    return contour_features_list

def featureExtraction(image):
  print(image,'xzodus')
  myImage = image
  image_resize = cv2.resize(image,(256,256),interpolation = cv2.INTER_AREA)
  # Color feature  # Find static data of HSV color
  hsv = cv2.cvtColor(image_resize,cv2.COLOR_RGB2HSV)
  payload_hsv = find_static_data(hsv,hsv = True)
  #Average
  H_mean = payload_hsv['mean'][0]
  S_mean = payload_hsv['mean'][1]
  V_mean = payload_hsv['mean'][2]
  #Standard deviation
  H_STD = payload_hsv['std'][0]
  S_STD = payload_hsv['std'][1]
  V_STD = payload_hsv['std'][2]
  # Skewness
  H_skewness = payload_hsv['skew'][0]
  S_skewness = payload_hsv['skew'][1]
  V_skewness = payload_hsv['skew'][2]
  # Kurtosis
  H_kurtosis = payload_hsv['kurtosis'][0]
  S_kurtosis = payload_hsv['kurtosis'][1]
  V_kurtosis = payload_hsv['kurtosis'][2]

  #   Shape feature extraction
  ec = calculate_eccentricity(image)
  eccentricity = ec

#   contour
  contour = extract_contour_features(myImage)
  print(contour)
  contourArea = contour[0]['Area']
  contourPerimeter = contour[0]['Perimeter']
  contourCompactness = contour[0]['Compactness']
  contourAspectRatio = contour[0]['Aspect Ratio']
  contourExtent = contour[0]['Extent']

#   Texture
  distances = [1]  # Distance between pixels in the GLCM
  angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles for GLCM computation
  haralick_features = compute_haralick_features(image_resize, distances, angles)
  print(haralick_features)
  Contrast = haralick_features['Contrast']
  Dissimilarity = haralick_features['Dissimilarity']
  Homogeneity = haralick_features['Homogeneity']
  Energy = haralick_features['Energy']
  Correlation = haralick_features['Correlation']

  data={
     'H_mean' : H_mean,
     'S_mean' : S_mean,
     'V_mean' : V_mean,
     'H_STD' : H_STD,
     'S_STD' : S_STD,
     'V_STD' : V_STD,
     'H_skewness' : H_skewness,
     'S_skewness' : S_skewness,
     'V_skewness' : V_skewness,
     'H_kurtosis' : H_kurtosis,
     'S_kurtosis' : S_kurtosis,
     'V_kurtosis' : V_kurtosis,
     'eccentricity': eccentricity,
     'contourArea' : contourArea,
     'contourPerimeter' : contourPerimeter,
     'contourCompactness' : contourCompactness,
     'contourAspectRatio' : contourAspectRatio,
     'contourExtent' : contourExtent,
     'Contrast' : Contrast,
     'Dissimilarity' : Dissimilarity,
     'Homogeneity' : Homogeneity,
     'Energy' : Energy,
     'Correlation' : Correlation,
  }

  print(data)

  return data



app = Flask(__name__)


@app.route("/")
def index():
    # return render_template('index.html', clear="True")
    return 'toon'


# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Function to check if the file extension is allowed
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # Route for uploading an image
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return 'No file part'

#     file = request.files['file']

#     if file.filename == '':
#         return 'No selected file'

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         return 'File uploaded successfully', 'filename', filename

#     return 'File type not allowed'




# # def predict():
# #     if request.files:
# #         file= request.files['image']
# #         print(file)
# #         # featureData=featureExtraction(file)
# #         return file
# #     else:
# #         return render_template('index.html', err="Have an error on image upload, Please try agian or contact associated department.")

# # def preprocess_image(path):
# #     image = tf.io.read_file(path)
# #     image = tf.image.decode_jpeg(image, channels=3)
# #     image = tf.image.convert_image_dtype(image, tf.float32)
# #     image = tf.image.resize(image, size=[224, 224])
# #     image = [np.array(image)]
# #     image = tf.data.Dataset.from_tensor_slices((tf.constant(image))).batch(32)

# #     return image


# # if __name__ != '__main__':
# #     gunicorn_logger = logging.getLogger('gunicorn.error')
# #     app.logger.handlers = gunicorn_logger.handlers
# #     app.logger.setLevel(gunicorn_logger.level)

# # if __name__ == "__main__":
#     app.run(host='0.0.0.0', debug=True)
