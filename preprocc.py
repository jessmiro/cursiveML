from PIL import Image
from PIL import ImageEnhance
import os
from os import listdir

#J: Preprocessing our data. Goes through file to pad it, brighten it, contrast it, resize it, and convert to grayscale
#J: we do this to prep our image for conversion to csv which is what's required for the training process
#J: and we want to avoid having too many different values.
#J: We used pillow to implement preprocessing.



#J: where my data is located, parsing through the images
folder_dir = "C:/Users/Jessica/Pictures/dataTest/"
for images in os.listdir(folder_dir):
 
    #J: check if the image ends with png, then we start the preprocessing process
    #J: basic code is from here https://www.geeksforgeeks.org/add-padding-to-the-image-with-python-pillow/
    #J: a few changes were made, including the if/else statement in order to pad to a square based on the largest measurement

    if (images.endswith(".png")):
        directory = folder_dir + "/" + images
        img = Image.open(directory).convert("RGB") #J: opening the image, we need to use .convert(RGB) so we don't just get a blob of color
        width, height = img.size #J: getting the width and height of the image
        height_split_r = 0 #J: initializing a few variables we're about to use
        height_split_l = 0
        width_split_r = 0
        width_split_l = 0

        #J: we need to add padding so that our image is a square. We go by the largest
        #measurement in order to create our square
        if(width > height):
            padding = width - height #J: how much padding we'll need
            height_split_r = padding/2 #J: padding around
            height_split_l = padding/2 #J: padding around
            height = height + height_split_l + height_split_r #J: our new height
        
        else:
            padding = height - width #J: how much padding we'll need
            width_split_r = padding/2 #J: padding around   
            width_split_l = padding/2 #J: padding around
            width = width + width_split_l + width_split_r #J: our new width
        
       #J: here is where we create a new image with the padding
       #J: where the padding is added around the original image
       #J: we pad using white. I didn't realize until post preprocessing that 249 wasn't the max
        result = Image.new(img.mode, (int(width), int(height)), (249, 249, 249))
        result.paste(img, (int(width_split_l), int(height_split_l)))
       
        #J: now we want to enhance the brightness of our image so the character is clearer
        enhancer1 = ImageEnhance.Brightness(result)
        factor1 = 1.25 #gives original image
        stuffs = enhancer1.enhance(factor1)

        #J: next we want to enhance the contrast of our image so the character is clearer
        #J: the brightness and contrast helped us go from yellow to white background
        #J: so there was no need to parse csv values and push them to black if it was at a certain value
        enhancer = ImageEnhance.Contrast(stuffs)
        factor = 2 #gives original image
        im_output = enhancer.enhance(factor)
       
       #J: finally we resize our image down to 28x28. I looked at a lot of handwriting recognizer code
       #J: and most people would have their data resized to 28x28
        resized = im_output.resize((28, 28))
       
       #J: Finally we convert to grayscale, so that our csv values are all in the same range
        resizer = resized.convert('LA')
        
        #J: where we save our images to
        os.chdir("C:/Users/Jessica/Pictures/dataTest/resized")
        resizer.save(str(images))
