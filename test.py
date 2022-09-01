import cv2 # computer vision library
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

'''
Function
'''
# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    
    ## TODO: Resize image and pre-process so that all "standard" images are the same size  
    copy_im = np.copy(image)
    
    # Define how many pixels to slice off the sides of the original image
    crop_percent = 10
    row_crop = int(copy_im.shape[1] * crop_percent /100)
    col_crop = int(copy_im.shape[0] * crop_percent /100)
    
    # Using image slicing, subtract the row_crop from top/bottom and col_crop from left/right   
    cropped_copy_im = image[row_crop:-row_crop, col_crop:-col_crop, :]
    standard_im = cv2.resize(cropped_copy_im, (32, 32))
    
    return standard_im

## TODO: One hot encode an image label
## Given a label - "red", "green", or "yellow" - return a one-hot encoded label

# Examples: 
# one_hot_encode("red") should return: [1, 0, 0]
# one_hot_encode("yellow") should return: [0, 1, 0]
# one_hot_encode("green") should return: [0, 0, 1]

def one_hot_encode(label):
    
    ## TODO: Create a one-hot encoded label that works for all classes of traffic lights

    if label =='red':
        one_hot_encoded = [1, 0, 0]
    elif label == 'yellow':
        one_hot_encoded =[0, 1, 0]
    else:
        one_hot_encoded = [0 ,0, 1]
        
    return one_hot_encoded

def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

'''
Execution
'''
# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)

## TODO: Write code to display an image in IMAGE_LIST (try finding a yellow traffic light!)
## TODO: Print out 1. The shape of the image and 2. The image's label

# The first image in IMAGE_LIST is displayed below (without information about shape or label)

#image number, i
i = 750

selected_image = IMAGE_LIST[i][0]
selected_image_label = IMAGE_LIST[i][1]
plt.imshow(selected_image)
print('Label:', selected_image_label)
print('Shape:', selected_image.shape)


print('done')

# Importing the tests
import test_functions
tests = test_functions.Tests()

# Test for one_hot_encode function
tests.test_one_hot(one_hot_encode)

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)

## TODO: Display a standardized image and its label
print('Standardized: ', STANDARDIZED_LIST[0][1])

print('Non-Standardized:',  IMAGE_LIST[0][1])

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('Standardized')
ax1.imshow(STANDARDIZED_LIST[0][0])
ax2.set_title('Non-Standardized')
ax2.imshow(IMAGE_LIST[0][0])

# Convert and image to HSV colorspace
# Visualize the individual color channels

image_num = 750
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

# Convert to HSV
hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

# Print image label
print('Label [red, yellow, green]: ' + str(test_label))

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

# Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(test_im)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')

## TODO: Create a brightness feature that takes in an RGB image and outputs a feature vector and/or value
## This feature should use HSV colorspace values

    
def create_hsv(rgb_image):
    
    ## TODO: Convert image to HSV color space
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    ## Add up all the pixel values and calculate the average brightness
    area =hsv.shape[0]*hsv.shape[1] #pixels
    
    #H channel
    h_sum_brightness = np.sum(hsv[:,:,0])
    h_avg = h_sum_brightness/area
    
    #S channel
    s_sum_brightness = np.sum(hsv[:,:,1])
    s_avg = s_sum_brightness/area
    
    #V channel
    v_sum_brightness = np.sum(hsv[:,:,2])
    v_avg = v_sum_brightness/area
    
    
    return h_avg, s_avg, v_avg

    
    ## TODO: Create and return a feature value and/or vector


def create_mask_image(rgb_image,label):
    
    #Convert image to HSV color space
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    #analyze histogram
    if label == 'red':
        red_mask1 = cv2.inRange(hsv, (0,30,50), (10,255,255))
        red_mask2 = cv2.inRange(hsv, (150,40,50), (180,255,255))
        mask = cv2.bitwise_or(red_mask1,red_mask2)
        
    elif label == 'yellow':
        mask = cv2.inRange(hsv, (10,10,110), (30,255,255))
    
    #green
    else:
        mask = cv2.inRange(hsv, (45,40,120), (95,255,255))
    
    res = cv2.bitwise_and(rgb_image,rgb_image,mask = mask)
    
    return res
    
def create_feature(rgb_image):

    h,s,v = create_hsv(rgb_image)
    image = np.copy (rgb_image)
    
    #apply mask
    red_mask = create_mask_image(image,'red')
    yellow_mask = create_mask_image(image,'yellow')
    green_mask = create_mask_image(image,'green')
    
    #slice into 3 parts, up, middle, down
    up = red_mask[0:10, :, :]
    middle = yellow_mask[11:20, :, :]
    down = green_mask[21:32, :, :]
    
    
    #find out hsv values based on each of the 3 parts
    
    h_up, s_up, v_up = create_hsv(up)
    h_middle, s_middle, v_middle = create_hsv(middle)
    h_down, s_down, v_down = create_hsv(down)
    
    #v in hsv can detect whether theres value in up,middle or down

    _
    if  v_up> v_middle and v_up> v_down:# and s_up>s_middle and s_up>s_down:
            
        return [1,0,0] #red

    elif  v_middle > v_down:# and s_middle>s_down:
        return [0,1,0] #yellow

    return [0,0,1] #green

