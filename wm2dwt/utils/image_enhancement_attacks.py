import cv2 as cv
import numpy as np
img_watermarked = cv.imread('example-images\\4.1.03.tiff')
print(img_watermarked.shape)
img_watermarked = cv.resize(img_watermarked, (512,512), interpolation=cv.INTER_CUBIC)
cv.imshow('Watermarked Image', img_watermarked)

img_watermarked = cv.resize(img_watermarked, (512,512), interpolation=cv.INTER_CUBIC)
(height, width) = img_watermarked.shape[:2]

# Bit-Plane Removal(x bits)
def bitPlaneRemoval(img, bits):
   temp_img= img
   for row in range(height):
      for column in range(width):
         blue, green, red = temp_img[row, column]
         n_blue = blue
         n_green = green
         n_red = red
         blue = (((n_blue >> bits) << bits) & blue)
         green = (((n_green >> bits) << bits) & green)
         red = (((n_red >> bits) << bits) & red)
         temp_img[row, column] = blue, green, red
   return temp_img

output_image_bitSlicing = bitPlaneRemoval(img_watermarked, 5)
cv.imshow('BitPlaneRemoved', output_image_bitSlicing)


# Gamma Correction
def gammaCorrection(img, gamma):
   temp_img = img
   temp_img = img.astype(np.float32) / 255.0
   for row in range(height):
      for column in range(width):
         blue, green, red = temp_img[row, column]
         blue = pow(blue, gamma)
         green = pow(green, gamma)
         red = pow(red, gamma)
         temp_img[row, column] = blue, green, red
   temp_img = (temp_img * 255).astype(np.uint8)     
   return temp_img

output_image_gammaCorrection1 = gammaCorrection(img_watermarked, 0.5)
cv.imshow('GammaCorrectedImage1', output_image_gammaCorrection1)
output_image_gammaCorrection = gammaCorrection(img_watermarked, 1.5)
cv.imshow('GammaCorrectedImage', output_image_gammaCorrection)

""" Histogram Equalization (old)
# def histogram_equalization(image):
#     image_eq = np.array(image, dtype = np.uint8)
#     histogram = np.zeros(256, dtype=np.int32)
#     for i in range(height):
#         for j in range(width):
#             histogram[image[i,j]]+=1
#     for i in range(1,256,1):         # Calculating the prefixSum or CDF
#         histogram[i]+=histogram[i-1]
#     for i in range(0,256,1):         # Normalizing the values 
#         histogram[i]/=histogram[255]
#     for i in range(0,256,1):         #
#         histogram[i]*=255
#     for i in range(0,256,1):
#         histogram[i]=round(histogram[i])
#     for i in range(n):
#         for j in range(m):
#             image_eq[i,j]=histogram[image[i,j]]
#     return image_eq


# histEqualized_img = histogram_equalization(img_watermarked)
# cv.imshow('HistEqualized_Image', histEqualized_img)
"""

def histogram_equalization(img):
   # convert the image from BGR to YCrCb color space
   ycrcb_img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
   # splitting the Y, Cr and Cb channels
   y, cr, cb = cv.split(ycrcb_img)
   # applying histogram equalization on Y channel
   y_eq = cv.equalizeHist(y)
   # merging the equalized Y channel back with Cr and Cb channels
   ycrcb_eq = cv.merge([y_eq, cr, cb])
   # converting the image back to BGR color space
   equalized_img = cv.cvtColor(ycrcb_img, cv.COLOR_YCrCb2BGR)
   return equalized_img

cv.imshow("hist_equ_img.jpg", histogram_equalization(img_watermarked))
# Laplacian Sharpening
def lapSharpening(img):
    blue, green, red = cv.split(img)
    laplacian_kernel = np.array([[1,1,1],
                                 [1,-8,1],
                                 [1,1,1]])
    conv_blue = cv.filter2D(blue, -1, laplacian_kernel, borderType=cv.BORDER_CONSTANT)
    conv_green = cv.filter2D(green, -1, laplacian_kernel, borderType=cv.BORDER_CONSTANT)
    conv_red = cv.filter2D(red, -1, laplacian_kernel, borderType=cv.BORDER_CONSTANT)
    filtered_image = cv.merge((conv_blue, conv_green, conv_red))
    sharpened_img = img - filtered_image
    return filtered_image

# sharpened_image = lapSharpening(img_watermarked)
# cv.imshow('SharpenedImage', sharpened_image)    
cv.waitKey(0)
cv.destroyAllWindows()