import numpy as np
""""""
# dct function
def dct(img:np.ndarray) -> np.ndarray:
    n = img.shape[0]
    l = np.arange(n)[:, np.newaxis]
    r = np.arange(n)
    
    cos_matrix = np.cos(np.pi * (2 * r + 1) * l / (2 * n))
    cos_matrix[0, :] *= 1 / np.sqrt(2)
    cos_matrix = np.sqrt(2 / n) * cos_matrix
    return np.dot(cos_matrix, img)

# inverse dct function
def idct(img:np.ndarray) -> np.ndarray:
    n = img.shape[0]
    l = np.arange(n)[:, np.newaxis]
    r = np.arange(n)
    
    cos_matrix = np.cos(np.pi * ((2 * r + 1) * l).T / (2 * n))
    
    cos_matrix[:, 0] *= 1 / np.sqrt(2)
    cos_matrix = np.sqrt(2 / n) * cos_matrix
    
    return np.dot(cos_matrix, img)

