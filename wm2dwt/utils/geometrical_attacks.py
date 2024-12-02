import cv2 as cv
import numpy as np

#Rotation(A custom rotate function which allows users to input the rotation angle )
def rotate_func(Y_new, rotation_angle, rotation_Point=None):
    (height, width) = Y_new.shape[:2]
    temp_img = np.copy(Y_new)
    if rotation_Point is None:
        rotation_Point = (width//2, height//2)
    rotation_Mat = cv.getRotationMatrix2D(rotation_Point, rotation_angle, 1.0)
    dimensions = (width, height)
    return cv.warpAffine(temp_img, rotation_Mat, dimensions)  

#Cropping surrounding pixel values by 15%
def cropping(Y_new):
    temp_img = np.copy(Y_new)
    (height, width) = Y_new.shape[:2]
    height_crop = int(0.15*height)
    width_crop = int(0.15*width)
    cropped_img = temp_img[height_crop: height-height_crop, width_crop:width-width_crop]
    return cropped_img

cv.waitKey(0)
cv.destroyAllWindows()