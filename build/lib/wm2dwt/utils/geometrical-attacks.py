import cv2 as cv
import numpy as np

img_watermarked = cv.imread('Images/Watermarked.jpg')
cv.imshow('Watermarked Image', img_watermarked)

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

cv.waitKey(0)
cv.destroyAllWindows()