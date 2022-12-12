import cv2 as cv
import csv
import numpy as np
import pandas as pd
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

#J: the code came from: https://github.com/Arnav1145/Handwritten-Character-Recognition/blob/main/Character%20recognition.ipynb
#J: Some pieces of code were omitted as it wasn't useful for my final result
#J: and some code was edited/added in order to correctly implement the model
#J: My data was broken down between two csv files, one that held the labels and one that held the csv values
#J: and this code was created based on only having one csv file. There's comments throughout where I explain what I changed and added
#J: Also most print statements are commented out as I printed as I went through the process

#J: This code deals with some additional preprocessing of the data and then the actual training model itself
#J: We split the data into training and testing. We also used keras to implement almost (if not) all of the model functions

#J: this file contains all of the csv values for the image data
data = pd.read_csv('C:/Users/Jessica/Pictures/dataTest/output_final.csv')
data.head()
data.info()
#print(data.shape)
my_data = data.values

#J: added. This file contains all of the labels for the image data. I used column 3 specifically
#J: because column 0 was the image file names, then the labels, then the labels converted from char to ascii, then modulo 97 so we could get much smaller values
#J: that also mapped nicely to the actual alphabet
df = pd.read_csv("C:/Users/Jessica/Desktop/labels_copy.csv", usecols=[3])
categorical_data = df.values
#print(categorical_data)

#J: creating the train and test datasets for the model
X = my_data[:,0:] #J: X holds the csv values
y = df.values[:,:1] # J: y needs to hold the labels
#print(X.shape)
#print(y.shape)
#J: Using 80% of the data for training and 20% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) #J: Since my data (csv values and labels) is split between two files, i couldn't randomize the data without losing the correct labels
X_train = np.reshape(X_train,(X_train.shape[0],28,28)) #J: reshaping the train images again
X_test = np.reshape(X_test,(X_test.shape[0],28,28)) #J: reshaping the test images again
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)


count = np.zeros(26, dtype = 'int') #count list containing all zeroes

#J: counting each label that appears in our data
#J: adding it to the count array
for i in y:
    item = i.item()
    count[item] += 1 


#creating a list of alphabets
#J: I changed it from upper to lowercase, since ~80%+ of my data was lowercase
#J: For future reference, I would remove the uppercase data from the datapool, as they're technically outliers
alphabets = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

 #J: This is just printing out the graph of quantity of each letter based on the labels
plt.figure(figsize=(15,10))
plt.barh(alphabets, count, color = "cyan")
plt.xlabel("Number of Alphabets",fontsize = 20, fontweight = 'bold',color = 'green')
plt.ylabel("Alphabets",fontsize = 30, fontweight = 'bold',color = 'green')
plt.title("No. of images available for each alphabet in the dataset", fontsize = 20, fontweight = 'bold', color = "red")
plt.grid()
plt.show()

#J: now we're just printing out a few pieces of our training list so we can see how it looks
img_list = X_train[:10]
print(img_list)
fig,ax = plt.subplots(3,3,figsize=(15,15))
axes = ax.flatten()
for i in range(9):
    axes[i].imshow(img_list[i])
    axes[i].grid()
plt.show()

#J: this was added by me so I could compare the images to their labels
imging = y_train[:10]
print(imging)

#J: We're reshaping our data once again
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2],1)

#J: checking the new shape
#print("New shape of train and test dataset")
#print(X_train.shape)
#print(X_test.shape)

#downsampling the values
X_train = X_train/255.
X_test = X_test/255.



        
#J: converting our label data to categorical data with 26 total classes (alphabet)
categorical_ytrain = to_categorical(y_train, num_classes = 26, dtype = 'int')
print("New shape of train labels:", categorical_ytrain.shape)

#J: converting our image data to categorical data with 26 total classes (alphabet)
categorical_ytest = to_categorical(y_test, num_classes = 26, dtype = 'int')
print("New shape of test labels:", categorical_ytest.shape)

#J: Here's where the actual model begins to start, and where we begin to train our data
# starting model
model = Sequential()

#First Conv1D layer
#J: our first layer. The filter size is 3x3. Using relu activation function. The input of our data is 28x28x1
#J: relu activation means rectified linear unit
model.add(Conv2D(32,kernel_size = (3,3),activation = 'relu',input_shape = (28,28,1)))
model.add(MaxPooling2D(pool_size = (2,2),strides = 2))

#Second Conv1D layer
#J: our second layer, we have 64 filters, the filter size is 3x3, activation function is relu, and we're padding with 0s (same)
model.add(Conv2D(filters = 64, kernel_size = (3,3),activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2,2), strides = 2))

#Third Conv1D layer
#J: Our file layer, we have 128 filters, filter size is still 3x3, activation function is still relu, but this time there's no padding
model.add(Conv2D(filters = 128, kernel_size = (3,3),activation = 'relu', padding = 'valid'))
model.add(MaxPooling2D(pool_size = (2,2), strides = 2))

#Flatten layer
model.add(Flatten())

#Dense layer 1
#J: dropout helps prevent overfitting
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))

#Dense layer 2
model.add(Dense(64,activation = 'relu'))

#Final layer of 26 nodes
#J: 26 nodes represents each letter of the alphabet
#J: softmax activation means we're converting a vector to probability distribution
model.add(Dense(26,activation = 'softmax'))

#Define the loss function to be categorical cross-entropy since it is a multi-classification problem:
#J: Using an optimizer to ensure out results, metrics are accuracy for the calculations of the predictions
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#J: early stopping is envoked in order to prevent overfitting
#J: modelcheckpoint deals with the weights and stores the best model in a .h5 file
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.001) 
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit(x = X_train, y = categorical_ytrain, epochs = 100, callbacks=[es,mc], validation_data = (X_test,categorical_ytest))

#J: Now we finally start training the model using the image csv values and the 
#J: categorical values of the labels
model.evaluate(X_test,categorical_ytest)

#J: prints out the summary of the model
model.summary()

#J: from here and onward, we're just printing out accuracies and losses, and plotting them
print("The validation accuracy is :", history.history['val_accuracy'][-1])
print("The training accuracy is :", history.history['accuracy'][-1])
print("The validation loss is :", history.history['val_loss'][-1])
print("The training loss is :", history.history['loss'][-1])

plt.figure(figsize = (6,6))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.title("Model Loss")
plt.show()

plt.figure(figsize = (6,6))
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.title("Model Accuracy")
plt.show()