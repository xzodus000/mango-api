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
model_file_path = os.path.join(os.getcwd(), 'Random_Forest_best_variety.pkl')
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

# def predict_mango(image_df):
#     try:
#         prediction = my_model.predict(image_df)
#         label_number = prediction[0]
        
#         # Check if the model supports probability predictions
#         if hasattr(my_model, "predict_proba"):
#             probabilities = my_model.predict_proba(image_df)
#             confidence = max(probabilities[0]) * 100  # Get highest probability
#             print(f'Prediction: {label_number} with {confidence:.2f}% confidence')
#         else:
#             print(f'Prediction: {label_number}')

#         return label_number  # You can also return confidence if needed
#     except Exception as e:
#         print("Error: Unable to make a prediction.")
#         print("Details:", str(e))
#         return None  # Return None if prediction fails


def predict_mango(image_df):
    try:
        # Make a prediction
        prediction = my_model.predict(image_df)
        label_number = prediction[0]
        
        # Ensure the predicted label is in known classes
        known_classes = my_model.classes_
        if label_number not in known_classes:
            print("Unknown Class Detected! Rejecting prediction.")
            return "Unknown Class"
        
        # Check if the model supports probability predictions
        if hasattr(my_model, "predict_proba"):
            probabilities = my_model.predict_proba(image_df)
            confidence = max(probabilities[0]) * 100  # Get highest probability
            
            # Apply confidence threshold (e.g., 50%)
            if confidence < 49:
                print(f"Low confidence ({confidence:.2f}%). Rejecting prediction.")
                return "UNKNOWN_CLASS"
            
            print(f'Prediction: {label_number} with {confidence:.2f}% confidence')
        else:
            print(f'Prediction: {label_number}')

        return label_number  
    except Exception as e:
        print("Error: Unable to make a prediction.")
        print("Details:", str(e))
        return None  # Return None if prediction fails


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

def extract_shape_features(image):
    """
    Extracts shape features from a given mango image for classification.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        dict: A dictionary containing the extracted shape features.
    """
    # Load the image
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"Error": "No contours found"}
    
    contour = max(contours, key=cv2.contourArea)  # Select the largest contour

    # Calculate shape features
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    compactness = area / (perimeter ** 2) if perimeter != 0 else 0
    rect_area = w * h
    extent = area / rect_area if rect_area != 0 else 0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area != 0 else 0

    # Fit an ellipse (if possible) for eccentricity
    if len(contour) >= 5:  # Minimum points required for ellipse fitting
        ellipse = cv2.fitEllipse(contour)
        (center, axes, orientation) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        eccentricity = np.sqrt(1 - (minor_axis ** 2 / major_axis ** 2))
    else:
        eccentricity = None

    # Calculate circularity
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

    # Compute Hu Moments
    hu_moments = cv2.HuMoments(cv2.moments(contour)).flatten()

    # Store features in a dictionary
  
    print(len(hu_moments.tolist()))

    features = {
        "Area": area,
        "Perimeter": perimeter,
        "Aspect_Ratio": aspect_ratio,
        "Compactness": compactness,
        "Extent": extent,
        "Solidity": solidity,
        "Eccentricity": eccentricity,
        "Circularity": circularity,
        **{f"Hu_Moment_{i+1}": moment for i, moment in enumerate(hu_moments)},
    }

    return features

def mango_extract_object(image: np.ndarray) -> np.ndarray:
    """
    Extracts the largest contour (assumed to be the mango) from the input image,
    then resizes it to fit within a 500x500 px canvas while preserving its aspect ratio.
    
    :param image: Input image as a NumPy array.
    :return: Processed 500x500 image with the extracted mango centered.
    """
    if image is None:
        raise ValueError("Error: Unable to read the image")
    
    # Convert to RGB and grayscale
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Error: No contours found in the image")
    
    # Select the largest contour (assuming it's the mango)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, color=255, thickness=cv2.FILLED)
    
    # Apply the mask to extract the mango object
    mango_object = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    
    # Get bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the mango object
    cropped_mango = mango_object[y:y+h, x:x+w]

    # Determine the aspect ratio
    aspect_ratio = w / h

    # Resize while maintaining aspect ratio
    if aspect_ratio > 1:
        # Wider than tall → width should be 500px
        new_w = 500
        new_h = int(500 / aspect_ratio)
    else:
        # Taller than wide → height should be 500px
        new_h = 500
        new_w = int(500 * aspect_ratio)

    resized_mango = cv2.resize(cropped_mango, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a black 500x500 canvas
    final_image = np.zeros((500, 500, 3), dtype=np.uint8)

    # Compute center position
    x_offset = (500 - new_w) // 2
    y_offset = (500 - new_h) // 2

    # Place the resized mango at the center of the canvas
    final_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_mango

    return final_image


