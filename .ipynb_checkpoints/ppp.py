import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
#load data
#miniproject='/mpdataset/'

no_tumor_images=os.listdir(R'C:\Users\91808\Desktop\mpdataset\no')
yes_tumor_images=os.listdir(R'C:\Users\91808\Desktop\mpdataset\yes')



dataset=[]
label=[]

INPUT_SIZE=64
#create lable
for i , image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        img_path1=os.path.join(R"C:\Users\91808\Desktop\mpdataset\no",image_name)
        image=cv2.imread(img_path1)
        image=Image.fromarray(image,'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        np.append(dataset,np.array(image))
        np.append(label,0)

        for i, image_name in enumerate(yes_tumor_images):
            if (image_name.split('.')[1] == 'jpg'):
                img_path2= os.path.join(R"C:\Users\91808\Desktop\mpdataset\yes", image_name)
                image = cv2.imread(img_path2)
                image = Image.fromarray(image, 'RGB')
                image = image.resize((INPUT_SIZE, INPUT_SIZE))
                np.append(dataset,np.array(image))
                np.append(label,1)

dataset = np.array(dataset)

label = np.array(label)

print('Dataset: ', len(dataset))
print('Label: ', len(label))
#train test split
X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=2023)
#normalize dataset
X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)
#model build
model=Sequential()

model.add(Conv2D(32, (3,3),activation='relu', input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))#relu will convert negative values to 0 and positive remains same
model.add(MaxPooling2D(pool_size=(2,2)))#maxpool reduces the size of matrix

model.add(Conv2D(32, (3,3),activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3),activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.summary()
# compile the model
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

# compile the model
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

#save the model
model.save('BrainTumorDetection.h5')

#make prediction on new data
def make_prediction(img):
    input_img = np.expand_dims(img, axis=0)

    res = (model.predict(input_img) > 0.5).astype("int32")
    return res


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

        #check tumor or not
        show_result('pred8.jpg')



"""import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
#load data
miniproject='mpdataset/'

no_tumor_images=os.listdir(miniproject+ 'no')
yes_tumor_images=os.listdir(miniproject+ 'yes')

print('No Tumor: ', len(no_tumor_images))
print('Tumor: ',len(yes_tumor_images))

dataset=[]
label=[]

INPUT_SIZE=64
#create lable
for i , image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(miniproject+'no/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

        for i, image_name in enumerate(yes):
            if (image_name.split('.')[1] == 'jpg'):
                image = cv2.imread(miniproject + 'yes/' + image_name)
                image = Image.fromarray(image, 'RGB')
                image = image.resize((INPUT_SIZE, INPUT_SIZE))
                dataset.append(np.array(image))
                label.append(1)

                dataset = np.array(dataset)

                label = np.array(label)

                print('Dataset: ', len(dataset))
                print('Label: ', len(label))
            #train test split
                X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=2023)
            #normalize dataset
            X_train = normalize(X_train, axis=1)
            X_test = normalize(X_test, axis=1)
        #model build
model=Sequential()

model.add(Conv2D(32, (3,3),activation='relu', input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))#relu will convert negative values to 0 and positive remains same
model.add(MaxPooling2D(pool_size=(2,2)))#maxpool reduces the size of matrix

model.add(Conv2D(32, (3,3),activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3),activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.summary()
# compile the model
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

# compile the model
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

#save the model
model.save('BrainTumorDetection.h5')

#make prediction on new data
def make_prediction(img):
    input_img = np.expand_dims(img, axis=0)

    res = (model.predict(input_img) > 0.5).astype("int32")
    return res


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

        #check tumor or not
        show_result('pred8.jpg')"""