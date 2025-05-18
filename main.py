import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image

from sklearn.model_selection import train_test_split

from keras.utils import normalize
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

# load data
miniproject = 'dataset/'

no_tumor_images = os.listdir(miniproject+'no')
yes_tumor_images = os.listdir(miniproject+'yes')

dataset = []
label = []

INPUT_SIZE = 64

# create label
for i, image_name in enumerate(no_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(miniproject+'no/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(miniproject+'yes/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

# train test split
X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=2023)

# normalize dataset
X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)

# model build
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# save the model
model.save('BrainTumorDetection.h5')

# make prediction on new data
def make_prediction(img):
   # img=normalize(img,axis=1)
    input_img = np.expand_dims(img, axis=0)
    return model.predict(input_img) > 0.5

def show_result(img):
    img_path = f"{miniproject}pred/{img}"
    image = cv2.imread(img_path)

    img = Image.fromarray(image)

    img = img.resize((64, 64))

    img = np.array(img)

    plt.imshow(img)
    plt.show()

    pred = make_prediction(img)
    
    if pred:
        print("Tumor Detected")
    else:
        print("No Tumor")

# check tumor or not
show_result('pred7.jpg')



