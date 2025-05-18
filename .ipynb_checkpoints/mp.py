import os
import numpy as np

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow_hub as hub


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
model = load_model(('BrainTumorDetection.h5'), custom_objects={'KerasLayer': hub.KerasLayer})


def model_predict(img_path, model):
    test_image = image.load_img(img_path, target_size = (224,224))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    result


    if result <= 0.5:
        result = "The Person has no Brain Tumor"
    else:
        result = "The Person has Brain Tumor"

    return result


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(_file_)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


if _name_ == '_main_':
    app.run(debug=False)

    < !DOCTYPE
    html >
    < html >
    < head >
    < title > Brain
    Tumor
    Detection < / title >
    < style >
    body
    {
        text - align: center;
    }

    h1
    {
        margin - top: 50px;
    }

    form
    {
        margin - top: 50px;
    }

    input[type = "submit"] {
        padding: 10px 20px;
    background - color:  # 4CAF50;
    color: white;
    border: none;
    cursor: pointer;
    }
    < / style >
< / head >
< body >
< h1 > Brain
Tumor
Detection < / h1 >
< form
action = "/predict"
method = "POST"
enctype = "multipart/form-data" >
< input
type = "file"
name = "file"
accept = "image/*"
required > < br > < br >
< input
type = "submit"
value = "Upload and Detect" >
< / form >
{ % if result %}
< h2 > {{result}} < / h2 >
{ % endif %}
{ % if image_path %}
< img
src = "{{ image_path }}"
alt = "Result Image"
style = "margin-top: 20px;" >
{ % endif %}
< / body >
< / html >