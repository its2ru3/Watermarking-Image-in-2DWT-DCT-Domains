import cv2 as cv
import numpy as np

# Bit-Plane Removal(x bits)
#this function removes last x bits from the original pixel values
def bitPlaneRemoval(Y_new, bits):
   temp_img = np.copy(Y_new)
   (height, width) = Y_new.shape[:2]
   for row in range(height):
      for column in range(width):
         temp_img[row, column] = (((temp_img[row, column] >> bits) << bits))
   return temp_img

# Gamma Correction
# It takes gamma values and does the correction as per the value
def gammaCorrection(Y_new, gamma):
   temp_img = np.copy(Y_new)
   temp_img = temp_img.astype(np.float32) / 255.0 # normalalizing the pixel values
   temp_img = pow(temp_img, gamma) # pow function raises the normalized pixel values to gamma
   temp_img = (temp_img * 255).astype(np.uint8)   # de-normalization of pixel values in the range of 0-255
   return temp_img

## Histogram Equalization (old)
def histogram_equalization(Y_new):
    temp_img = np.copy(Y_new)
    image_eq = np.array(temp_img, dtype = np.uint8)
    histogram = np.zeros(256, dtype=np.int32)
    (height, width) = Y_new.shape[:2]
    for i in range(height):
        for j in range(width):
            histogram[temp_img[i,j]]+=1
    for i in range(1,256,1):         # Calculating the prefixSum or CDF
        histogram[i]+=histogram[i-1]
    for i in range(0,256,1):         # Normalizing the values 
        histogram[i]/=histogram[255]
    for i in range(0,256,1):         
        histogram[i]*=255
    for i in range(0,256,1):
        histogram[i]=round(histogram[i])
    for i in range(height):
        for j in range(width):
            image_eq[i,j]=histogram[temp_img[i,j]] # mapping the pixel values to the new values
    return image_eq


"""# histEqualized_img = histogram_equalization(img_watermarked)
# cv.imshow('HistEqualized_Image', histEqualized_img)


# def histogram_equalization(img):
#    # convert the image from BGR to YCrCb color space
#    ycrcb_img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
#    # splitting the Y, Cr and Cb channels
#    y, cr, cb = cv.split(ycrcb_img)
#    # applying histogram equalization on Y channel
#    y_eq = cv.equalizeHist(y)
#    # merging the equalized Y channel back with Cr and Cb channels
#    ycrcb_eq = cv.merge([y_eq, cr, cb])
#    # converting the image back to BGR color space
#    equalized_img = cv.cvtColor(ycrcb_img, cv.COLOR_YCrCb2BGR)
#    return equalized_img """

# Laplacian Sharpening
def lapSharpening(Y_new):
    temp_img = np.cop(Y_new)
    laplacian_kernel = np.array([[1,1,1], # 
                                 [1,-8,1],
                                 [1,1,1]])
    filtered_img = cv.filter2D(temp_img, -1, laplacian_kernel, borderType=cv.BORDER_CONSTANT)
    sharpened_img = Y_new - filtered_img # adding the original image with the filtered image to get the background details as well as sharpened values on the final output image
    return sharpened_img
  
cv.waitKey(0)
cv.destroyAllWindows()