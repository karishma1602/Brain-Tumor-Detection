from flask import Flask, render_template, request
import cv2
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

from keras.utils import normalize
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('BrainTumorDetection.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Get the uploaded tumor file
    tumor_file = request.files['tumor']
    tumor_filename = tumor_file.filename
    tumor_path = os.path.join('uploads', tumor_filename)
    tumor_file.save(tumor_path)

    # Perform tumor detection and get the result
    tumor_detected = show_result(tumor_path)

    # Render the HTML template with the result
    return render_template('index.html', tumor_filename=tumor_filename, tumor_detected=tumor_detected)

def show_result(img_path):
    image = cv2.imread(img_path)
    img = Image.fromarray(image)
    img = img.resize((64, 64))
    img = np.array(img)
    img = normalize(img, axis=1)

    plt.imshow(img)
    plt.show()

    pred = make_prediction(img)
    return pred

def make_prediction(img):
   # img=normalize(img,axis=1)
    input_img = np.expand_dims(img, axis=0)
    return model.predict(input_img)[0][0] > 0.5

if __name__ == '__main__':
    app.run(debug=True)
