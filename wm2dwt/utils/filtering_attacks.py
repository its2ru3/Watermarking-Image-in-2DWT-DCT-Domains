import cv2 as cv
import numpy as np

# Average Filter (mask size = 3X3)
# here kernel(size = 3X3) convolves with the original image. 
# cv.filter2D function is used for convolution. Pixel values are initially normalized and then converted back in the range 0-255
def avgFilter(Y_new):
    temp_img = np.copy(Y_new) # copy of original image is converted to avoid changes in it.
    temp_img = temp_img/255.0
    kernel = np.array([[1/9,1/9,1/9],
                       [1/9,1/9,1/9],
                       [1/9,1/9,1/9]])
    temp_img = cv.filter2D(temp_img, -1, kernel, borderType=cv.BORDER_CONSTANT) # cv.BORDER_CONSTANT pads the image with constant value
    temp_img = (temp_img*255).astype(np.uint8) # to convert the pixel values in the range 0-255 for showing using openCV
    return temp_img

# Median Filter (mask size = 3X3)
#  
def medianFilter(Y_new):
    temp_img = np.copy(Y_new) 
    (h,w) = temp_img.shape[:2]
    filtered_img = np.zeros([h,w], dtype=np.uint8) # filtered image is initiallized with zeros
    temp_img = np.pad(temp_img, 1, mode='constant', constant_values=0)  # padding of temp_img is done here
    for row in range(1, h-1): # Iterating the temp_img in the neighbourhood of 3X3
        for col in range(1, w-1): 
            temp = [temp_img[row-1, col-1], # a temp numpy array stores the neighbourhood values (in 3X3)
               temp_img[row-1, col], 
               temp_img[row-1, col + 1], 
               temp_img[row, col-1], 
               temp_img[row, col], 
               temp_img[row, col + 1], 
               temp_img[row + 1, col-1], 
               temp_img[row + 1, col], 
               temp_img[row + 1, col + 1]]          
            temp=np.sort(temp)   
            filtered_img[row, col]= temp[4]                
    filtered_img = filtered_img.astype(np.uint8)
    return filtered_img

# Gaussian Filter (mask size = 3X3)
def gaussianFilter1(Y_new):
    temp_img = np.copy(Y_new)
    temp_img = temp_img.astype(np.float32)/255.0 # normalizing the pixel values
    kernel = np.array([[1/16,2/16,1/16],    # kernel of size 3X3
                       [2/16,4/16,2/16],
                       [1/16,2/16,1/16]])
    temp_img = cv.filter2D(temp_img, -1, kernel, borderType=cv.BORDER_CONSTANT) # convolution of kernel with the temp_img 
    temp_img = (temp_img*255).astype(np.uint8) 
    return temp_img

# Gaussian Filter (mask size = 5X5)
def gaussianFilter2(Y_new):
    temp_img = np.copy(Y_new)
    temp_img = temp_img.astype(np.float32)/255.0 
    kernel = np.array([[1/273,4/273,6/273,4/273,1/273], # kernel of size 5X5
                       [4/273,16/273,24/273,16/273,4/273],
                       [6/273,24/273,36/273,24/273,6/273],
                       [4/273,16/273,24/273,16/273,4/273],
                       [1/273,4/273,6/273,4/273,1/273]])
    temp_img = cv.filter2D(temp_img, -1, kernel, borderType=cv.BORDER_CONSTANT)
    temp_img = (temp_img*255).astype(np.uint8)
    return temp_img

cv.waitKey(0)
cv.destroyAllWindows()