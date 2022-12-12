from PIL import Image
import numpy as np
import sys
import os
import csv

#J: This file just deals with converting from greyscale to csv values, and then saving it into one document

#J: The original code is from Pam's response in this stackoverflow question: https://stackoverflow.com/questions/49070242/converting-images-to-csv-file-in-python 
#J: a few lines were removed but nothing was added other than changing the image format, location of the original images, and where to write to

#J: Here we're just creating an array that stores all of the files
def createFileList(myDir, format='.png'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

#J: load the original image, this is where my resized images are located
myFileList = createFileList('C:/Users/Jessica/proj/df_resized')

#J: now we parse through the array in order to compile and convert to a csv
for file in myFileList:
    print(file)
    img_file = Image.open(file)


    # get original image parameters...
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    # Make image Greyscale
    img_grey = img_file.convert('L')

    # Save Greyscale values
    value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
    value = value.flatten()
    print(value)

    #J: we're saving all of the csv / greyscale values into output_final.csv
    with open("output_final.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)