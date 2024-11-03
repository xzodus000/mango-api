# app.py

import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from services import allowed_file, calculate_eccentricity, find_static_data_lab_and_rgb, extract_contour_features, predict_by_variety, compute_haralick_features, find_static_data, predict_mango
import cv2
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_file():
    try:
        if 'file' not in request.files:
            logger.error('No file part')
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if not file or not allowed_file(file.filename):
            logger.error('Invalid file extension')
            return jsonify({'error': 'Invalid file extension'}), 400
        
        file_data = file.read()
        nparr = np.frombuffer(file_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_resize = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

        df = pd.DataFrame({'eccentricity': [calculate_eccentricity(image)]})
        contour_features = extract_contour_features(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        
        if contour_features:
            contour = contour_features[0]
            df['contourArea'] = contour['Area']
            df['contourPerimeter'] = contour['Perimeter']
            df['contourCompactness'] = contour['Compactness']
            df['contourAspectRatio'] = contour['Aspect Ratio']
            df['contourExtent'] = contour['Extent']

        haralick_features = compute_haralick_features(image_resize, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])
        for key, value in haralick_features.items():
            df[key] = value

        hsv_image = cv2.cvtColor(image_resize, cv2.COLOR_RGB2HSV)
        static_data = find_static_data(hsv_image, hsv=True)
        
        df['H_mean'], df['S_mean'], df['V_mean'] = static_data['mean']
        df['H_STD'], df['S_STD'], df['V_STD'] = static_data['std']
        df['H_skewness'], df['S_skewness'], df['V_skewness'] = static_data['skew']
        df['H_kurtosis'], df['S_kurtosis'], df['V_kurtosis'] = static_data['kurtosis']

        df = df.reindex(sorted(df.columns), axis=1)
        label = predict_mango(df)

        return jsonify({'statusCode': 200, 'data': label})

    except Exception as e:
        logger.error(f'Error processing file: {e}')
        return jsonify({'error': 'An error occurred while processing the file', 'details': str(e)}), 500

@app.route('/upload-phase-maturity', methods=['POST'])
@cross_origin()
def upload_file_phase_2():
    try:
        if 'file' not in request.files:
            logger.error('No file part')
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if not file or not allowed_file(file.filename):
            logger.error('Invalid file extension')
            return jsonify({'error': 'Invalid file extension'}), 400
        
        file_data = file.read()
        logger.info('Reading image and resizing...')
        nparr = np.frombuffer(file_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error('Image is None, check the file format and data.')
            return jsonify({'error': 'Uploaded file is not a valid image'}), 400
        
        image_resize = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

        df = pd.DataFrame({'eccentricity': [calculate_eccentricity(image)]})
        contour_features = extract_contour_features(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        
        if contour_features:
            contour = contour_features[0]
            df['contourArea'] = contour['Area']
            df['contourPerimeter'] = contour['Perimeter']
            df['contourCompactness'] = contour['Compactness']
            df['contourAspectRatio'] = contour['Aspect Ratio']
            df['contourExtent'] = contour['Extent']

        haralick_features = compute_haralick_features(image_resize, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])
        for key, value in haralick_features.items():
            df[key] = value

        hsv_image = cv2.cvtColor(image_resize, cv2.COLOR_RGB2HSV)
        static_data = find_static_data(hsv_image, hsv=True)
        
        df['H_mean'], df['S_mean'], df['V_mean'] = static_data['mean']
        df['H_STD'], df['S_STD'], df['V_STD'] = static_data['std']
        df['H_skewness'], df['S_skewness'], df['V_skewness'] = static_data['skew']
        df['H_kurtosis'], df['S_kurtosis'], df['V_kurtosis'] = static_data['kurtosis']

        df = df.reindex(sorted(df.columns), axis=1)
        label = predict_mango(df)

        image_resize_500 = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)

         
        if label == "Mahachanok":
            # Assuming this block is executed
            rgb_image = image_resize_500
            static_data_RGB = find_static_data_lab_and_rgb(rgb_image)
            logger.info(f'Static data for Mahachanok: {static_data_RGB}')
            if static_data_RGB is not None:
                df_maturity = pd.DataFrame({'R_mean': [static_data_RGB['mean'][0]]})
                # pd.DataFrame({'R_mean': [static_data_RGB['mean'][0]]})
                df_maturity['R_mean'] = static_data_RGB['mean'][0]
                df_maturity['G_mean'] = static_data_RGB['mean'][1]
                df_maturity['B_mean'] = static_data_RGB['mean'][2]
                df_maturity['R_STD'] = static_data_RGB['std'][0]
                df_maturity['G_STD'] = static_data_RGB['std'][1]
                df_maturity['B_STD'] = static_data_RGB['std'][2]
                df_maturity['R_skewness'] = static_data_RGB['skew'][0]
                df_maturity['G_skewness'] = static_data_RGB['skew'][1]
                df_maturity['B_skewness'] = static_data_RGB['skew'][2]
                df_maturity['R_kurtosis'] = static_data_RGB['kurtosis'][0]
                df_maturity['G_kurtosis'] = static_data_RGB['kurtosis'][1]
                df_maturity['B_kurtosis'] = static_data_RGB['kurtosis'][2]
        elif label in ["Nam dok mai", "R2E2"]:
            # Populate df_maturity here for other labels
            lab_image = cv2.cvtColor(image_resize, cv2.COLOR_BGR2LAB)
            static_data_LAB = find_static_data_lab_and_rgb(lab_image)
            logger.info(f'Static data for Nam dok mai / R2E2: {static_data_LAB}')
            print('toonLable',static_data_LAB['mean'][0])
            if static_data_LAB is not None:
                df_maturity = pd.DataFrame({'L_mean': [static_data_LAB['mean'][0]]})
                df_maturity['L_mean'] = static_data_LAB['mean'][0]
                df_maturity['A_mean'] = static_data_LAB['mean'][1]
                df_maturity['B_mean'] = static_data_LAB['mean'][2]
                df_maturity['L_STD'] = static_data_LAB['std'][0]
                df_maturity['A_STD'] = static_data_LAB['std'][1]
                df_maturity['B_STD'] = static_data_LAB['std'][2]
                df_maturity['L_skewness'] = static_data_LAB['skew'][0]
                df_maturity['A_skewness'] = static_data_LAB['skew'][1]
                df_maturity['B_skewness'] = static_data_LAB['skew'][2]
                df_maturity['L_kurtosis'] = static_data_LAB['kurtosis'][0]
                df_maturity['A_kurtosis'] = static_data_LAB['kurtosis'][1]
                df_maturity['B_kurtosis'] = static_data_LAB['kurtosis'][2]

                print('xzodus',df_maturity)
        
        logger.info(f'df_maturity before prediction: {df_maturity}')
        logger.info(f'df_maturity shape: {df_maturity.shape}')
        if df_maturity.empty:
            logger.error('df_maturity is still empty after attempting to populate it.')
            return jsonify({'error': 'No data available for prediction'}), 400
        
        df_maturity = df_maturity.reindex(sorted(df_maturity.columns), axis=1)

        predicted_result = predict_by_variety(df_maturity, label)
        print('toon res predict',predicted_result)

        return jsonify({'statusCode': 200, 'data': predicted_result})

    except Exception as e:
        logger.error(f'Error processing phase maturity file: {e}')
        return jsonify({'error': 'An error occurred while processing the file', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
