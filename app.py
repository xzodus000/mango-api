from flask import Flask, request, jsonify
import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
app = Flask(__name__)
import pandas as pd
import cv2
import numpy as np
from io import BytesIO
from collections import Counter
from scipy.stats import skew,kurtosis
import os
import pickle
from skimage.feature import graycomatrix, graycoprops
# Set the upload folder and allowed extensions for images
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_file_path = os.path.join(os.getcwd(), 'RandomForestClassifiermodel.pkl')
my_model = pickle.load(open(model_file_path,'rb'))

mango_array_label = [
    {"name": "MHN", "value": 0},
    {"name": "NDM", "value": 1},
    {"name": "R2E0", "value": 2}
]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function for calculating LBP
def lbp_calculated_pixel(img, x, y):
   
    center = img[x][y]
   
    val_ar = []
      
    # top_left
    val_ar.append(get_pixel(img, center, x-1, y-1))
      
    # top
    val_ar.append(get_pixel(img, center, x-1, y))
      
    # top_right
    val_ar.append(get_pixel(img, center, x-1, y + 1))
      
    # right
    val_ar.append(get_pixel(img, center, x, y + 1))
      
    # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))
      
    # bottom
    val_ar.append(get_pixel(img, center, x + 1, y))
      
    # bottom_left
    val_ar.append(get_pixel(img, center, x + 1, y-1))
      
    # left
    val_ar.append(get_pixel(img, center, x, y-1))
       
    # Now, we need to convert binary
    # values to decimal
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
   
    val = 0
      
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
          
    return val

def get_pixel(img, center, x, y):
      
    new_value = 0
      
    try:
        # If local neighbourhood pixel 
        # value is greater than or equal
        # to center pixel values then 
        # set it to 1
        if img[x][y] >= center:
            new_value = 1
              
    except:
        # Exception is required when 
        # neighbourhood value of a center
        # pixel value is null i.e. values
        # present at boundaries.
        pass
      
    return new_value

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

def extract_contour_features(image):
    # Load the image
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

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    data = {'label': [''],}  # Replace with your own data
   
    file_data = file.read()
    nparr = np.frombuffer(file_data, np.uint8)

    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_resize = cv2.resize(image,(256,256),interpolation = cv2.INTER_AREA)


    row=0
    ec = calculate_eccentricity(image)

    data = {'eccentricity': [ec]}
    df = pd.DataFrame(data)
    contour = extract_contour_features(gray_image)

    df.loc[row,'contourArea'] = contour[0]['Area']
    df.loc[row,'contourPerimeter'] = contour[0]['Perimeter']
    df.loc[row,'contourCompactness'] = contour[0]['Compactness']
    df.loc[row,'contourAspectRatio'] = contour[0]['Aspect Ratio']
    df.loc[row,'contourExtent'] = contour[0]['Extent']




      # # # Texture Feature Extraction Haralick
    distances = [1]  # Distance between pixels in the GLCM
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles for GLCM computation
    haralick_features = compute_haralick_features(image_resize, distances, angles)
    print(haralick_features)
    df.loc[row,'Contrast'] = haralick_features['Contrast']
    df.loc[row,'Dissimilarity'] = haralick_features['Dissimilarity']
    df.loc[row,'Homogeneity'] = haralick_features['Homogeneity']
    df.loc[row,'Energy'] = haralick_features['Energy']
    df.loc[row,'Correlation'] = haralick_features['Correlation']




    # #lbp
    # height, width, _ = image_resize.shape
    # img_gray = cv2.cvtColor(image_resize,cv2.COLOR_BGR2GRAY)
    # img_lbp = np.zeros((height, width),np.uint8)
    # for i in range(0, height):
    #     for j in range(0, width):
    #         img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
    # vector_lbp = img_lbp.flatten()
    # counted = Counter(vector_lbp)
    # for key,value in counted.items():
    #     df.loc[row, f"lbp_85"] = 0
    #     df.loc[row, f"lbp_{key}"] = value


    # #   # Find static data of HSV color
    hsv = cv2.cvtColor(image_resize,cv2.COLOR_RGB2HSV)
    payload_hsv = find_static_data(hsv,hsv = True)

    #Average
    df.loc[row,'H_mean'] = payload_hsv['mean'][0]
    df.loc[row,'S_mean'] = payload_hsv['mean'][1]
    df.loc[row,'V_mean'] = payload_hsv['mean'][2]
    #Standard deviation
    df.loc[row,'H_STD'] = payload_hsv['std'][0]
    df.loc[row,'S_STD'] = payload_hsv['std'][1]
    df.loc[row,'V_STD'] = payload_hsv['std'][2]
    # Skewness
    df.loc[row,'H_skewness'] = payload_hsv['skew'][0]
    df.loc[row,'S_skewness'] = payload_hsv['skew'][1]
    df.loc[row,'V_skewness'] = payload_hsv['skew'][2]
    # Kurtosis
    df.loc[row,'H_kurtosis'] = payload_hsv['kurtosis'][0]
    df.loc[row,'S_kurtosis'] = payload_hsv['kurtosis'][1]
    df.loc[row,'V_kurtosis'] = payload_hsv['kurtosis'][2]
    
    # df
    # df
    print(df)
    print(os.getcwd())
    print(file.filename)
    df = df.reindex(sorted(df.columns), axis=1)
    

    
    # df.to_csv('my_image_data.csv',index=False)
    # if list(X_test.columns) != list(X_train.columns):
    #     print()
    # X = df.drop("label",axis=1)
    # y = df["label"]
    # X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=1.0)
    mango = my_model.predict(df)
    print('toon-mango')
    print(type(mango))
    print(mango[0])
    lableNumber = mango[0]
    print(type(lableNumber))
    if len(mango) == []:
        print("Array is empty")
    else:
        print("Array is not empty")
        filtered_array = list(filter(lambda x: x.get("value") == lableNumber, mango_array_label))
        print(filtered_array)
        lableNumber=filtered_array[0].get("name")

    
    # loaded_model = joblib.load('RandomForestClassifiermodel.joblib', mmap_mode=None)
    # pickled_model = pickle.load(open('RandomForestClassifiermodel.pkl', 'rb'))
    # pickled_model = pickle.load(open('RandomForestClassifiermodel.pkl', 'rb'))




    # loaded_model.

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        # Save the uploaded file to the uploads folder
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

        # You can perform additional processing here, such as storing the file path in a database

        return jsonify({'statusCode': 200, 'data' : str(lableNumber)})

    return jsonify({'error': 'Invalid file extension'})

if __name__ == '__main__':
    app.run(debug=True)
