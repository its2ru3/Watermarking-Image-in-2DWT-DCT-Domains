import cv2 as cv
import numpy as np
img_watermarked = cv.imread('Images/yv.jpg')
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


# Histogram Equalization
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

sharpened_image = lapSharpening(img_watermarked)

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

(height, width) = img_watermarked.shape[:2]

#Rotation
def rotate_func(img_watermarked, rotation_angle, rotation_Point=None):

    if rotation_Point is None:
        rotation_Point = (width//2, height//2)

    rotation_Mat = cv.getRotationMatrix2D(rotation_Point, rotation_angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img_watermarked, rotation_Mat, dimensions)  

rotated_image = rotate_func(img_watermarked, 45)
cv.imshow('Rotated', rotated_image)

#Cropping surrounding pixel values by 15%
height_crop = int(0.15*height)
width_crop = int(0.15*width)
cropped_img = img_watermarked[height_crop: height-height_crop, width_crop:width-width_crop]
cv.imshow('CroppedImage', cropped_img)

# Resizing Image to different dimensions to 256 & 200
img_resized_256 = cv.resize(img_watermarked, (256,256), interpolation=cv.INTER_AREA)
cv.imshow('IMG256', img_resized_256)
img_resized_200 = cv.resize(img_watermarked, (200, 200), interpolation=cv.INTER_AREA)
cv.imshow('IMG200', img_resized_200)


## JPEG Compression
# Define the JPEG Compression quality (0-100, where 100 is the highest quality)
compression_quality = 80

# Save the image after JPEG compression
cv.imwrite("compressed_image_2.jpg", img_watermarked, [cv.IMWRITE_JPEG_QUALITY, compression_quality])

# Gaussian Noise
def gaussian_noise_adder(img, std_dev, mean=0):
    noise = np.random.normal(mean, std_dev, img.shape)
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

cv.imshow('Noisy(gaussian- var = 0.0005) Image', gaussian_noise_adder(img_watermarked,  0.0005))
cv.imshow('Noisy(gaussian- var = 0.001) Image', gaussian_noise_adder(img_watermarked,  0.001))


# Salt and pepper noise
def salt_and_pepper_noise_adder(img, std_dev, mean = 0):
    noise = np.random.saltan(mean, std_dev, img.shape)
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

cv.imshow('Noisy(gaussian- var = 0.0005) Image', gaussian_noise_adder(img_watermarked,  0.0005))
cv.imshow('Noisy(gaussian- var = 0.001) Image', gaussian_noise_adder(img_watermarked,  0.001))
