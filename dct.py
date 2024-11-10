import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def scale(image):
    current_max = np.max(image)
    current_min = np.min(image)
    scaled_image = ((image-current_min) / (current_max-current_min)) * 255
    return scaled_image


# dct function
def dct(array):
    n = array.shape[0]
    l = np.arange(n)[:, np.newaxis]
    r = np.arange(n)
    
    cos_matrix = np.cos(np.pi * (2 * r + 1) * l / (2 * n))
    
    cos_matrix[0, :] *= 1 / np.sqrt(2)
    
    cos_matrix = np.sqrt(2 / n) * cos_matrix
    
    return np.dot(cos_matrix, array)


# inverse dct function
def idct(array):
    n = array.shape[0]
    l = np.arange(n)[:, np.newaxis]
    r = np.arange(n)
    
    cos_matrix = np.cos(np.pi * ((2 * r + 1) * l).T / (2 * n))
    
    cos_matrix[:, 0] *= 1 / np.sqrt(2)
    cos_matrix = np.sqrt(2 / n) * cos_matrix
    
    return np.dot(cos_matrix, array)

