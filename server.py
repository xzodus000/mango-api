# app.py

import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from services import allowed_file,mango_extract_object,extract_shape_features, calculate_eccentricity, find_static_data_lab_and_rgb, extract_contour_features, predict_by_variety, compute_haralick_features, find_static_data, predict_mango
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
# Load your trained model
# model = load_model("your_model.h5")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route('/upload', methods=['POST'])
@cross_origin()
def pre_upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if not file or not allowed_file(file.filename):
        logger.error('Invalid file extension')
        return jsonify({'error': 'Invalid file extension'}), 400
        
    file_data = file.read()
    nparr = np.frombuffer(file_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is not None:
        h, w = image.shape[:2]
        print(f"Width: {w}, Height: {h}")
    else:
        print("Failed to decode image.")
    image = mango_extract_object(image)
    image_resize = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)

 
    contour_features = extract_shape_features(image_resize)
   
        

    #     contour = contour_features[0]
    df = pd.DataFrame({
        'Compactness': [contour_features['Compactness']],
        'Extent': [contour_features['Extent']],
        'Solidity': [contour_features['Solidity']],
        'Eccentricity': [contour_features['Eccentricity']]
    })


    df = df.reindex(sorted(df.columns), axis=1)
    #  return contour_features
 
    label = predict_mango(df)
    print(label)

    return jsonify({'statusCode': 200, 'data': label})


@app.route('/upload-phase-maturity', methods=['POST'])
@cross_origin()
def phase_upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if not file or not allowed_file(file.filename):
        logger.error('Invalid file extension')
        return jsonify({'error': 'Invalid file extension'}), 400
        
    file_data = file.read()
    nparr = np.frombuffer(file_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Print the width and height of the image from file_data
    if image is not None:
        h, w = image.shape[:2]
        print(f"Width: {w}, Height: {h}")
    else:
        print("Failed to decode image.")
    image = mango_extract_object(image)
    image_resize = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)

    contour_features = extract_shape_features(image_resize)
   
    #     contour = contour_features[0]
    df = pd.DataFrame({
        'Compactness': [contour_features['Compactness']],
        'Extent': [contour_features['Extent']],
        'Solidity': [contour_features['Solidity']],
        'Eccentricity': [contour_features['Eccentricity']]
    })


    df = df.reindex(sorted(df.columns), axis=1)
    #  return contour_features
 
    label = predict_mango(df)
    print(label)

    model_MHN = tf.keras.models.load_model("model-maturity/MHN.h5")
    model_NDM = tf.keras.models.load_model("model-maturity/NDM.h5")
    model_R2E2 = tf.keras.models.load_model("model-maturity/R2E2.h5")
    model = model_MHN

    match label:
        case "MHN":
            model = model_MHN
        case "NDM":
            model = model_NDM
        case "R2E2":
            model = model_R2E2



    file_data = file.read()
    nparr = np.frombuffer(file_data, np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image shape before resizing: {image.shape}")
    image_resize = cv2.resize(image, (128, 128))
    print(f"Image shape after resizing: {image_resize.shape}")
    image_resize = np.expand_dims(image_resize, axis=0)  # Add batch dimension
    print(f"Final input shape: {image_resize.shape}")  # Should be (1, 128, 128, 3)
    prediction = model.predict(image_resize)

    # Interpret results
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the highest probability class
    confidence = np.max(prediction)  # Get confidence score

    # Print results
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")
    
    res = f'{label}_M{int(predicted_class) + 1}'
    return jsonify({'statusCode': 200, 'data': res})





if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
