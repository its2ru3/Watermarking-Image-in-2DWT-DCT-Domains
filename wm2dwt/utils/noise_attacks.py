import cv2 as cv
import numpy as np

# Gaussian Noise
def gaussian_noise_adder(img, std_dev, mean=0): 
    noise = np.random.normal(mean, std_dev, img.shape)
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

# Salt and pepper noise
def salt_and_pepper_noise_adder(img, std_dev, mean = 0):
    noise = np.random.saltan(mean, std_dev, img.shape)
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

cv.waitKey(0)
cv.destroyAllWindows()