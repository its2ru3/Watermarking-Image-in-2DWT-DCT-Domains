import numpy as np
import pywt
import os
from PIL import Image
from scipy.fftpack import dct, idct
from scipy.ndimage import convolve
# Get the current path of the script
alpha=1
current_path = str(os.path.dirname(__file__))

# File names for the input image and watermark
image = 'krishna.jpeg'   
watermark = 'Untitled.png' 

# Convert an image to grayscale, resize it, and return as a NumPy array
def convert_image(image_name, size):
    # Open and resize the image
    img = Image.open('./example-images/' + image_name).resize((size, size), 1)
    # Convert to grayscale
    img = img.convert('L')
    # Save the processed image
    img.save('./dataset/' + image_name)

    # Convert the image to a NumPy array
    image_array = np.array(img.getdata(), dtype=np.float64).reshape((size, size))
    print(image_array[0][0])                # Print the first pixel value
    print(image_array[10][10])              # Print a random pixel value

    return image_array

# Perform 2D Discrete Wavelet Transform (DWT) and return coefficients
def process_coefficients(imArray, model, level):
    coeffs = pywt.wavedec2(data=imArray, wavelet=model, level=level)
    coeffs_H = list(coeffs)  # Convert to a mutable list
    return coeffs_H

# Embed the watermark into the coefficients using mod2 embedding
def embed_mod2(coeff_image, coeff_watermark, offset=0):
    for i in range(len(coeff_watermark)):
        for j in range(len(coeff_watermark[i])):
            coeff_image[i*2+offset][j*2+offset] = coeff_watermark[i][j]
    return coeff_image

# Embed the watermark into the coefficients using mod4 embedding
def embed_mod4(coeff_image, coeff_watermark):
    for i in range(len(coeff_watermark)):
        for j in range(len(coeff_watermark[i])):
            coeff_image[i*4][j*4] = coeff_watermark[i][j]
    return coeff_image

# Embed the watermark into the DCT coefficients of the original image
def embed_watermark(watermark_array, orig_image):
    watermark_flat = watermark_array.ravel()  # Flatten the watermark array
    ind = 0

    # Iterate over the image in 8x8 blocks
    for x in range(0, len(orig_image), 8):
        for y in range(0, len(orig_image), 8):
            if ind < len(watermark_flat):
                subdct = orig_image[x:x+8, y:y+8]
                subdct[5][5] = alpha*watermark_flat[ind]  # Embed watermark at [5][5] in each block
                orig_image[x:x+8, y:y+8] = subdct
                ind += 1

    return orig_image

# Apply 2D DCT on 8x8 blocks of an image
def apply_dct(image_array):
    size = len(image_array)
    all_subdct = np.empty((size, size))
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            subpixels = image_array[i:i+8, j:j+8]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")  # Apply 2D DCT
            all_subdct[i:i+8, j:j+8] = subdct
    return all_subdct

# Apply 2D inverse DCT on 8x8 blocks
def inverse_dct(all_subdct):
    size = len(all_subdct)
    all_subidct = np.empty((size, size))
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            subidct = idct(idct(all_subdct[i:i+8, j:j+8].T, norm="ortho").T, norm="ortho")  # Apply 2D IDCT
            all_subidct[i:i+8, j:j+8] = subidct
    return all_subidct

# Extract the watermark from the DCT coefficients
def get_watermark(dct_watermarked_coeff, watermark_size):
    subwatermarks = []
    for x in range(0, len(dct_watermarked_coeff), 8):
        for y in range(0, len(dct_watermarked_coeff), 8):
            coeff_slice = dct_watermarked_coeff[x:x+8, y:y+8]
            subwatermarks.append(coeff_slice[5][5])  # Extract the embedded value from [5][5]
    watermark = np.array(subwatermarks).reshape(watermark_size, watermark_size)
    return watermark

# Recover the watermark from the watermarked image
def recover_watermark(image_array, model='haar', level=1):
    coeffs_watermarked_image = process_coefficients(image_array, model, level=level)
    dct_watermarked_coeff = apply_dct(coeffs_watermarked_image[0])
    watermark_array = get_watermark(dct_watermarked_coeff, 128)
    watermark_array = np.uint8(watermark_array)

    # Save the recovered watermark
    img = Image.fromarray(watermark_array)
    img.save('./result/recovered_watermark.jpg')

# Save an image from a NumPy array
def print_image_from_array(image_array, name):
    image_array_copy = image_array.clip(0, 255)  # Clip values to valid range
    image_array_copy = image_array_copy.astype("uint8")
    img = Image.fromarray(image_array_copy)
    img.save('./result/' + name)



def gaussianFilter(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * np.exp(
            - ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)
        ),
        (size, size),
    )
    return kernel / np.sum(kernel)

# Main function to apply watermark embedding and recovery
def w2d(img):
    model = 'haar'
    level = 1

    # Convert and preprocess images
    image_array = convert_image(image, 2048)
    watermark_array = convert_image(watermark, 128)

    # Process image coefficients
    coeffs_image = process_coefficients(image_array, model, level=level)
    dct_array = apply_dct(coeffs_image[0])

    # Embed the watermark into the DCT coefficients
    dct_array = embed_watermark(watermark_array, dct_array)
    coeffs_image[0] = inverse_dct(dct_array)

    # Reconstruct the watermarked image
    image_array_H = pywt.waverec2(coeffs_image, model)

    print_image_from_array(image_array_H, 'image_with_watermark.jpg')

    size = 25
    sigma = 30
    gaussian_filter = gaussianFilter(size, sigma)

# Step 3: Apply the filter to the image using convolution
    Y_atks = convolve(image_array_H, gaussian_filter)
    # print(type(image_array))
    y=convolve(image_array, gaussian_filter)
    print_image_from_array(y,'blur_orig.jpg')
    print_image_from_array(Y_atks,'attacked_image.jpg')

    recover_watermark(image_array=image_array_H, model=model, level=level)

# Run the watermark embedding and recovery
w2d("test")