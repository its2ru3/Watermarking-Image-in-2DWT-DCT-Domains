import cv2
def get_grayscale(image):
    """ Convert to GrayScale
    The grayscale version of the image is a single-channel image,
    created by taking a weighted sum of the R, G, and B values.
    Gray = 0.299×R + 0.587×G + 0.114×B
    gray_image is a numpy array in unit8 format """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image
    
