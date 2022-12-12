import os
import natsort

#J: I had the images on my macbook and my code on a windows laptop
#J: so we had to first do some editing to get the images to appear the same 
filenames = os.listdir('datafiles')
filenames.remove(".DS_Store") #J: get rid of the mac os files
filenames = natsort.natsorted(filenames) #J: sort like mac os folder does
#print(filenames)

#J: Opening up a .csv file to write to and the .txt file to read from
fileoutput = open("labels.csv", "w")
fileinput = open("labels.txt", "r")

lines = fileinput.readlines()

#J: first we need to rename the image files
final_list = []
count = 0
for line in lines:
  #J: create new filenames based on data instead of screenshot time
  final_list.append([line.strip(), filenames[count], str(str(count).zfill(4) + "_" + line.strip() + ".png")])
  count += 1

os.chdir("datafiles")

#J: now we're going through and writing the new image name, a comma, and then
#J: the correct label
for f in final_list:
  fileoutput.write(str(f[2]) + "," + str(f[0]  + "\n"))

