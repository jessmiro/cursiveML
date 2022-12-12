# cursiveML
handwriting recognition with cursive data



This repository is for my 486 final project, handwriting recognition using cursive characters.

Since I wanted to see how my model compared to already existing work and because I was very new at machine learning and python, I used pre-existing code to create my
model. Because of this, my own comments include "#J:" as I didn't want to remove the original comments.

Links to the code I used:
Images to CSV: https://stackoverflow.com/questions/49070242/converting-images-to-csv-file-in-python
Preprocessing Padding: https://www.geeksforgeeks.org/add-padding-to-the-image-with-python-pillow/
Handwriting recognition model using tensorflow: https://github.com/Arnav1145/Handwritten-Character-Recognition/blob/main/Character%20recognition.ipynb




**Files:**
preprocc.py: Preproccessing my data. Padded the image to a square, increased brightness, increased contrast, converted to grayscale, and resize to 28x28 pixels.
Preprocessing was done so that we could have smaller grayscale value matrices to work with. But before that, we needed to make sure the data was all the same size
(padding and resizing) and that it was easy to see where the characters began/stopped (brightness, contrast, grayscale).

png_to_csv.py: Converting the images (.png) to csv values, and then storing it to a single csv document.

modelTest.py: This is where the model lives. I have some additional preproccessing/pre-model code in the same folder. Tensorflow + keras was used to implement the CNN
model. I also made quite a few changes to the original code as that code assumed a single csv file containing the image matrices and the image labels. My data was split 
between a csv file containing just the matrices and another csv containing the image names and the labels. 

dataTest folder: Contains my original image data. There's around 1800 images, all of cursive characters. Data collection was done by hand, as I felt that scraping using 
code would take much longer and provide less quality data. The data in this file is post-parced for unreadable data (Original dataset size was 2000 images).

df_resized folder: Contains the preprocessed and resized image data.

output_final.csv: The matrices corresponding to the image data.

labels_copy.csv: Contains four columns: image names, labels, labels in ASCII format, and labels post %97 for much more usable corresponding char values

best_model.h5: Result of running the code






This code can be run during different intervals at a time. 

To start from the beginning, download just the dataTest folder and replace the location in preprocc.py. Then use the results of that program to run png_to_csv.py after 
replacing the location, and then use the results to run modelTest.py. Make sure to replace the location in the code and also download the labels_copy.csv and replace 
the location in the code. 

To begin from png_to_csv.py, you need to download the df_resized folder and replace the location in the code, and then use those results and labels_copy.csv and replace the locations in the code.

To begin with modelTest.py, you need to download labels_copy.csv and output_final.csv and replace the locatiosn in the code.
