import numpy as np

def psnr(I1: np.ndarray, I2: np.ndarray) -> float:
    # Check if the sizes of I1 and I2 are the same
    if I1.shape != I2.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    mse = np.mean((I1 - I2) ** 2)
    if mse == 0:
        return float('inf')  # PSNR is infinite if the images are identical
    
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value


def ssim(I1: np.ndarray, I2: np.ndarray) -> float:
    # Ensure images have the same dimensions
    if I1.shape != I2.shape:
        raise ValueError("Input images must have the same dimensions.")
    mu1 = np.mean(I1)
    mu2 = np.mean(I2)
    sigma1_sqr = np.var(I1)
    sigma2_sqr = np.var(I2)

    covariance_matrix = np.cov(I1.flatten(), I2.flatten())
    # Extract the covariance value between I1 and I2
    sigma12 = covariance_matrix[0, 1]

    # Constants for stability
    # C1 and C2 are constants that ensure stability when the denominator becomes 0
    # Commonly, C1=(K1×L)^2 and C2=(K2×L)^2, where L is the dynamic range of the pixel values
    # (e.g., 255 for 8-bit images), and K1​ and K2​ are small constants (e.g., 0.01 and 0.03)
    K1 = 0.01
    K2 = 0.03
    L = 255 # for 8-bit grey-scale image
    C1 = (K1*L)**2
    C2 = (K2*L)**2

    # SSIM calculation
    SSIM = ( (2*mu1*mu2 + C1) * (2*sigma12 + C2) ) / ( (mu1**2 + mu2**2 + C1) * (sigma1_sqr + sigma2_sqr + C2) )
    SSIM = np.clip(SSIM,0,1) # Ensure SSIM is within [0, 1]
    print(mu1, mu2, sigma1_sqr, sigma2_sqr, sigma12)
    print(SSIM)
    return SSIM 

def bcr(W: np.ndarray, W_: np.ndarray) -> float:
    # Ensure W and W_ have the same length
    if W.shape != W_.shape:
        raise ValueError("Input arrays must have the same length.")
    
    z = len(W)
    BCR = 0
    for i in range(z):
        if(W[i]==W_[i]):
            BCR += 1

    BCR_percentage = (BCR * 100) / z  # Convert to percentage
    return BCR_percentage

# # Example usage
# z = 128
# W = np.random.choice([0, 1], size=z)
# W_ = np.random.choice([0, 1], size=z)

# print(W)
# print(W_)
# print("BCR:", bcr(W, W_), "%")
