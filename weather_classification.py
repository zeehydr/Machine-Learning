# Load the Drive helper and mount
from google.colab import drive
# This will prompt for authorization.
drive.mount('/content/drive')
# After executing the cell above, Drive files will be present in "/content/drive/My
Drive".
!ls "/content/drive/My Drive"
# Important imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import random
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing import image
#from keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.utils import img_to_array, array_to_img
from keras.optimizers import Adam
from PIL import Image
from keras.models import Sequential
#from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense,
LeakyReLU,AveragePooling2D
from sklearn.model_selection import train_test_split
convolutional neural network (CNN). ResNet50 is a deep learning model trained on more than a
million images from the ImageNet dataset. It's a popular choice for image classification tasks and is
known for its ability to achieve high accuracy while also being able to handle very deep network
architectures. ResNet50 is trained to identify 1000 classes of objects, the architecture of the model is
based on "residual connections" which allows the model to learn even deeper representation of the
image.
Proposed System
The proposed system focuses on how to identify the weather from different classes by using
RESNET50 with the help of computer vision and deep learning algorithm by using the Tensorflow,
Keras, numpy, matplotlib.pyplot and pandas library.
1. Python Code
# Listing directory
!ls "/content/drive/My Drive/WeatherImages"
# Plotting 25 images to check dataset
plt.figure(figsize=(11,11))
path = "/content/drive/My Drive/WeatherImages/Rainbow"
for i in range(1,26):
plt.subplot(5,5,i)
plt.tight_layout()
rand_img = imread(path +'/'+ random.choice(sorted(listdir(path))))
plt.imshow(rand_img)
plt.title('Rainbow')
plt.xlabel(rand_img.shape[1], fontsize = 10)
plt.ylabel(rand_img.shape[0], fontsize = 10)
# Check number of images for each type of weather condition
num_cloudy = len(os.listdir(r'C:\Users\HP\Desktop\WeatherImages/Cloudy/'))
num_rain = len(os.listdir(r'C:\Users\HP\Desktop\WeatherImages/Rain/'))
num_shine = len(os.listdir(r'C:\Users\HP\Desktop\WeatherImages/Shine/'))
num_sunrise = len(os.listdir(r'C:\Users\HP\Desktop\WeatherImages/Sunrise/'))
num_fogsmog = len(os.listdir(r'C:\Users\HP\Desktop\WeatherImages/Fogsmog/'))
num_rainbow = len(os.listdir(r'C:\Users\HP\Desktop\WeatherImages/Rainbow/'))
num_Cirroculumulus =
len(os.listdir(r'C:\Users\HP\Desktop\WeatherImages/Cirroculumulus/'))
num_Cirrostratus =
len(os.listdir(r'C:\Users\HP\Desktop\WeatherImages/Cirrostratus/'))
num_Cirrus = len(os.listdir(r'C:\Users\HP\Desktop\WeatherImages/Cirrus/'))
num_Cumulonimbus =
len(os.listdir(r'C:\Users\HP\Desktop\WeatherImages/Cumulonimbus/'))
num_Cumulus = len(os.listdir(r'C:\Users\HP\Desktop\WeatherImages/Cumulus/'))
#classes = ["Cirroculumulus", "Cirrostratus", "Cirrus", "Cloudy",
"Cumulonimbus","Cumulus","Fogsmog","Rain","Rainbow","Shine","Sunrise"]
# Plot distribution of classes
def label_pie(pct, allvals):
absolute = int(round(pct/100.*np.sum(allvals)))
return "{:.1f}%\n{:d} images".format(pct, absolute)
fig = plt.figure(figsize=(6,6))
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
weather_conditions = ["Cirroculumulus", "Cirrostratus", "Cirrus", "Cloudy",
"Cumulonimbus","Cumulus","Fogsmog","Rain","Rainbow","Shine","Sunrise"]
num_images =
[num_cloudy,num_rain,num_shine,num_sunrise,num_fogsmog,num_rainbow,num_Cirroculumul
us,num_Cirrostratus,num_Cirrus,num_Cumulonimbus,num_Cumulus]
ax.pie(num_images, labels = weather_conditions, autopct=lambda pct: label_pie(pct,
num_images), textprops={'fontsize': 8})
plt.title('Composition of original dataset ({} total
images)'.format(sum(num_images)), fontsize=15)
plt.show()
# Setting root directory path and creating empty list
dir = "/content/drive/My Drive/WeatherImages"
root_dir = listdir(dir)
image_list, label_list = [], []
# Reading and converting image to numpy array
for directory in root_dir:
for files in listdir(f"{dir}/{directory}"):
image_path = f"{dir}/{directory}/{files}"
image = Image.open(image_path)
image = image.resize((150,150)) # All images does not have same dimension
image = img_to_array(image)
image_list.append(image)
label_list.append(directory)
# Visualize the number of classes count
label_counts = pd.DataFrame(label_list).value_counts()
label_counts
# Checking count of classes
num_classes = len(label_counts)
num_classes
# Checking x data shape
np.array(image_list).shape
# Checking y data shape
label_list = np.array(label_list)
label_list.shape
# Splitting dataset into test and train
x_train, x_test, y_train, y_test = train_test_split(image_list, label_list,
test_size=0.2, random_state = 10)
# Normalize and reshape data
x_train = np.array(x_train, dtype=np.float16) / 225.0
x_test = np.array(x_test, dtype=np.float16) / 225.0
x_train = x_train.reshape( -1, 150,150,3)
x_test = x_test.reshape( -1, 150,150,3)
# Binarizing labels
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)
print(lb.classes_)
# Splitting the training data set into training and validation data sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size =
0.2)
from tensorflow.keras.applications.resnet50 import ResNet50
# Get the ResNet50 base model
basemodel = ResNet50(weights = 'imagenet', include_top = False, input_tensor =
Input(shape=(150, 150, 3)))
basemodel.summary()
# freeze the model weights
for layer in basemodel.layers:
layers.trainable = False
headmodel = basemodel.output
headmodel = AveragePooling2D(pool_size = (4,4))(headmodel)
headmodel = Flatten(name= 'flatten')(headmodel)
headmodel = Dense(256, activation = "relu")(headmodel)
headmodel = Dropout(0.3)(headmodel)#
headmodel = Dense(256, activation = "relu")(headmodel)
headmodel = Dropout(0.3)(headmodel)
#headmodel = Dense(256, activation = "relu")(headmodel)
#headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(11, activation = 'softmax')(headmodel)
model = Model(inputs = basemodel.input, outputs = headmodel)
model.summary()
# Compiling model
model.compile(loss = 'categorical_crossentropy', optimizer =
Adam(0.0005),metrics=['accuracy'])
# Training the model
epochs = 25
batch_size = 64
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs,
validation_data = (x_val, y_val))
#Plot the training history
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], color='r')
plt.plot(history.history['val_accuracy'], color='b')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'val'])
plt.show()
#Plot the loss history
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], color='r')
plt.plot(history.history['val_loss'], color='b')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'val'])
plt.show()
# Calculating test accuracy
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")
# Storing model predictions
y_pred = model.predict(x_test)
# Plotting image to compare
img = array_to_img(x_test[6])
img
Results
The results of our project are listed below;
⚫ The desired output on 25 epochs 
# Finding max value from predition list and comaparing original value vs predicted
labels = lb.classes_
print(labels)
print("Originally : ",labels[np.argmax(y_test[6])])
print("Predicted : ",labels[np.argmax(y_pred[6])])
# Saving model
model.save("/content/drive/My Drive/intel_image.h5")