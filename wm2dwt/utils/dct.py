import numpy as np

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


# DCT function
# def dct(img: np.ndarray) -> np.ndarray:
#     """
#     Perform the 1D Discrete Cosine Transform (DCT) on an input array.
    
#     Args:
#         img (np.ndarray): Input 1D array.
    
#     Returns:
#         np.ndarray: Transformed array.
#     """
#     n = img.shape[0]
#     l = np.arange(n)[:, np.newaxis]
#     r = np.arange(n)
    
#     # Construct the cosine transform matrix
#     cos_matrix = np.cos(np.pi * (2 * r + 1) * l / (2 * n))
#     cos_matrix[0, :] *= 1 / np.sqrt(2)  # Scale first row (DC component)
#     cos_matrix = np.sqrt(2 / n) * cos_matrix  # Overall scaling
    
#     # Perform DCT
#     return np.dot(cos_matrix, img)

# # Inverse DCT function
# def idct(img: np.ndarray) -> np.ndarray:
#     """
#     Perform the 1D Inverse Discrete Cosine Transform (IDCT) on an input array.
    
#     Args:
#         img (np.ndarray): Input 1D array.
    
#     Returns:
#         np.ndarray: Inverse transformed array.
#     """
#     n = img.shape[0]
#     l = np.arange(n)[:, np.newaxis]
#     r = np.arange(n)
    
#     # Construct the inverse cosine transform matrix
#     cos_matrix = np.cos(np.pi * (2 * r + 1) * l / (2 * n))
#     cos_matrix[0, :] *= 1 / np.sqrt(2)  # Scale first column (DC component)
#     cos_matrix = np.sqrt(2 / n) * cos_matrix  # Overall scaling
    
#     # Perform IDCT
#     return np.dot(cos_matrix.T, img)
