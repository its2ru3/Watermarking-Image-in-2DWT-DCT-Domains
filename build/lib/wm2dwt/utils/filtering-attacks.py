import cv2 as cv
import numpy as np
img_watermarked = cv.imread('Images/Flower.jpg')
img_watermarked = cv.resize(img_watermarked, (512,512), interpolation=cv.INTER_CUBIC)
cv.imshow('Watermarked Image', img_watermarked)

# Average Filter (mask size = 3X3)
def avgFilter(img):
    img = img.astype(np.float32)/255.0
    blue, green, red = cv.split(img)
    kernel = np.array([[1/9,1/9,1/9],
                       [1/9,1/9,1/9],
                       [1/9,1/9,1/9]])
    conv_blue = cv.filter2D(blue, -1, kernel, borderType=cv.BORDER_CONSTANT)
    conv_green = cv.filter2D(green, -1, kernel, borderType=cv.BORDER_CONSTANT)
    conv_red = cv.filter2D(red, -1, kernel, borderType=cv.BORDER_CONSTANT)
    filtered_image = cv.merge((conv_blue, conv_green, conv_red))
    filtered_image = (filtered_image*255).astype(np.uint8)
    return filtered_image
filtered_img = avgFilter(img_watermarked)
cv.imshow('AvgFilteredImg', filtered_img)


# Median Filter (mask size = 3X3)
def medianFilter(img):
    (h,w) = img.shape[:2]
    blue, green, red = cv.split(img)
    np.pad(blue, 1, mode='constant', constant_values=0)
    np.pad(blue, 1, mode='constant', constant_values=0)
    np.pad(blue, 1, mode='constant', constant_values=0)

    n_blue = np.zeros([h,w])
    n_green = np.zeros([h,w])
    n_red = np.zeros([h,w])
    for row in range(1, h-1): 
        for col in range(1, w-1): 
            temp = [blue[row-1, col-1], 
               blue[row-1, col], 
               blue[row-1, col + 1], 
               blue[row, col-1], 
               blue[row, col], 
               blue[row, col + 1], 
               blue[row + 1, col-1], 
               blue[row + 1, col], 
               blue[row + 1, col + 1]]          
            temp=np.sort(temp)   
            n_blue[row, col]= temp[4] 

    for row in range(1, h-1): 
        for col in range(1, w-1): 
            temp = [green[row-1, col-1], 
               green[row-1, col], 
               green[row-1, col + 1], 
               green[row, col-1], 
               green[row, col], 
               green[row, col + 1], 
               green[row + 1, col-1], 
               green[row + 1, col], 
               green[row + 1, col + 1]]           
            temp=np.sort(temp) 
            n_green[row, col]= temp[4]

    for row in range(1, h-1): 
        for col in range(1, w-1): 
            temp = [red[row-1, col-1], 
               red[row-1, col], 
               red[row-1, col + 1], 
               red[row, col-1], 
               red[row, col], 
               red[row, col + 1], 
               red[row + 1, col-1], 
               red[row + 1, col], 
               red[row + 1, col + 1]]          
            temp=np.sort(temp)   
            n_red[row, col]= temp[4]                
    filtered_image = cv.merge((n_blue, n_green, n_red))
    filtered_image = filtered_image.astype(np.uint8)
    return filtered_image

filtered_img = medianFilter(img_watermarked)
cv.imshow('MedianFilteredImg', filtered_img)


# Gaussian Filter (mask size = 3X3)
def gaussianFilter1(img):
    img = img.astype(np.float32)/255.0
    blue, green, red = cv.split(img)
    kernel = np.array([[1/16,2/16,1/16],
                       [2/16,4/16,2/16],
                       [1/16,2/16,1/16]])
    conv_blue = cv.filter2D(blue, -1, kernel, borderType=cv.BORDER_CONSTANT)
    conv_green = cv.filter2D(green, -1, kernel, borderType=cv.BORDER_CONSTANT)
    conv_red = cv.filter2D(red, -1, kernel, borderType=cv.BORDER_CONSTANT)
    filtered_image = cv.merge((conv_blue, conv_green, conv_red))
    filtered_image = (filtered_image*255).astype(np.uint8)
    return filtered_image
filtered_img = gaussianFilter1(img_watermarked)
cv.imshow('Gauss3FilteredImg', filtered_img)


# Gaussian Filter (mask size = 5X5)
def gaussianFilter2(img):
    img = img.astype(np.float32)/255.0
    blue, green, red = cv.split(img)
    kernel = np.array([[1/273,4/273,6/273,4/273,1/273],
                       [4/273,16/273,24/273,16/273,4/273],
                       [6/273,24/273,36/273,24/273,6/273],
                       [4/273,16/273,24/273,16/273,4/273],
                       [1/273,4/273,6/273,4/273,1/273]])
    conv_blue = cv.filter2D(blue, -1, kernel, borderType=cv.BORDER_CONSTANT)
    conv_green = cv.filter2D(green, -1, kernel, borderType=cv.BORDER_CONSTANT)
    conv_red = cv.filter2D(red, -1, kernel, borderType=cv.BORDER_CONSTANT)
    filtered_image = cv.merge((conv_blue, conv_green, conv_red))
    filtered_image = (filtered_image*255).astype(np.uint8)
    return filtered_image
filtered_img = gaussianFilter2(img_watermarked)
cv.imshow('Gauss5FilteredImg', filtered_img)


cv.waitKey(0)
cv.destroyAllWindows()