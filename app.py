import os
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__,template_folder="templates")

# Load the trained model
# with open('brain_tumor_model.pkl','rb') as file:
#     loaded_model=pickle.load(file)
model=joblib.load(R"C:\Users\91808\Desktop\mpdataset\BrainT.pkl")

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file from the request
        file = request.files['file']

        # Save the file to the uploads folder
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))

        # Load and preprocess the image for prediction
        img = image.load_img(os.path.join('uploads', filename), target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        # Make the prediction
        prediction = model.predict(img)
        result = "The Person has Brain Tumor" if prediction[0][0] > 0.5 else "The Person has no Brain Tumor"

        # Prepare the response as JSON
        response = {
            'result': result,
            'image_path': f'uploads/{filename}'
        }

        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)