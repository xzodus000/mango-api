# services.py

import os
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
import pickle
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Load the trained model
model_file_path = os.path.join(os.getcwd(), 'RandomForestClassifiermodel.pkl')
my_model = pickle.load(open(model_file_path, 'rb'))

# Labels for mango varieties
mango_array_label = [
    {"name": "Mahachanok", "value": 0},
    {"name": "Nam dok mai", "value": 1},
    {"name": "R2E2", "value": 2}
]

mango_array_phase_2_MHN_label = [
    {"name": "MHN_M1", "value": 0},
    {"name": "MHN_M1", "value": 1},
    {"name": "MHN_M1", "value": 2}
]

mango_array_phase_2_NDM_label = [
    {"name": "NDM_M1", "value": 0},
    {"name": "NDM_M1", "value": 1},
    {"name": "NDM_M1", "value": 2}
]

mango_array_phase_2_R2E2_label = [
    {"name": "R2E2_M1", "value": 0},
    {"name": "R2E2_M1", "value": 1},
]

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def calculate_eccentricity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)
    major_axis = 2 * ((moments["mu20"] + moments["mu02"] + ((moments["mu20"] - moments["mu02"])**2 + 4 * (moments["mu11"]**2))**0.5) / moments["m00"])**0.5
    minor_axis = 2 * ((moments["mu20"] + moments["mu02"] - ((moments["mu20"] - moments["mu02"])**2 + 4 * (moments["mu11"]**2))**0.5) / moments["m00"])**0.5
    return major_axis / minor_axis

def extract_contour_features(image):
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_features = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        compactness = (perimeter ** 2) / area if area != 0 else 0.0
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        extent = area / (w * h)
        contour_features.append({
            "Area": area,
            "Perimeter": perimeter,
            "Compactness": compactness,
            "Aspect Ratio": aspect_ratio,
            "Extent": extent
        })
    return contour_features

def compute_haralick_features(image, distances, angles):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, distances, angles, levels=256, symmetric=True, normed=True)
    
    # Use the correct property names
    features = {
        "Contrast": np.mean(graycoprops(glcm, 'contrast')),
        "Dissimilarity": np.mean(graycoprops(glcm, 'dissimilarity')),
        "Homogeneity": np.mean(graycoprops(glcm, 'homogeneity')),
        "Energy": np.mean(graycoprops(glcm, 'energy')),
        "Correlation": np.mean(graycoprops(glcm, 'correlation'))
    }
    
    return features

def find_static_data(img, hsv=False):
    resized_image = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    mean = resized_image.mean(axis=(0, 1))
    sd = resized_image.std(axis=(0, 1))
    
    if hsv:
        h, s, v = cv2.split(img)
        payload = {
            'mean': mean,
            'std': sd,
            'skew': [skew(h.flatten()), skew(s.flatten()), skew(v.flatten())],
            'kurtosis': [kurtosis(h.flatten()), kurtosis(s.flatten()), kurtosis(v.flatten())]
        }
    else:
        b, g, r = cv2.split(img)
        payload = {
            'mean': mean,
            'std': sd,
            'skew': [skew(b.flatten()), skew(g.flatten()), skew(r.flatten())],
            'kurtosis': [kurtosis(b.flatten()), kurtosis(g.flatten()), kurtosis(r.flatten())]
        }
    
    return payload

# Color Feature extraction Hsv and Rgb
def find_static_data_lab_and_rgb(img,hsv=False):
  #b,g,r
  #h,s,v
  # It calculates the mean and standard deviation along the specified axes (0 and 1) of the input image
  mean = img.mean(axis=(0,1))
  sd = img.std(axis=(0,1))
  if hsv:
    h,s,v = cv2.split(img)
    # The Hue, Saturation, and Value channels are flattened into 1D arrays. Then, skewness (skew) and kurtosis (kurtosis) are calculated for each channel.
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
    # The Blue, Green, and Red channels are flattened into 1D arrays, and skewness and kurtosis are calculated for each channel.
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

def predict_mango(image_df):
    prediction = my_model.predict(image_df)
    label_number = prediction[0]
    filtered_label = next((item["name"] for item in mango_array_label if item["value"] == label_number), "Unknown")
    return filtered_label

def predict_by_variety(image_df, variety):
    models = {
    "Mahachanok": pickle.load(open(os.path.join(os.getcwd(), 'model','phase2','MHN','Classifier_best_model.pkl'), 'rb')),
    "Nam dok mai": pickle.load(open(os.path.join(os.getcwd(), 'model','phase2','NDM','Classifier_best_model.pkl'), 'rb')),
    "R2E2": pickle.load(open(os.path.join(os.getcwd(), 'model','phase2','R2E2','Classifier_best_model.pkl'), 'rb'))
    }

    if variety not in models:
        raise ValueError(f"Model for variety '{variety}' not available.")

    model = models[variety]
    
    # Run the prediction
    prediction = model.predict(image_df)
    label_number = prediction[0]
    if variety == 'Mahachanok':
        mango_label = mango_array_phase_2_MHN_label
    elif variety == "Nam dok mai":
        mango_label = mango_array_phase_2_NDM_label
    elif variety == "R2E2":
        mango_label = mango_array_phase_2_R2E2_label

    filtered_label = next((item["name"] for item in mango_label if item["value"] == label_number), "Unknown")
    return filtered_label
