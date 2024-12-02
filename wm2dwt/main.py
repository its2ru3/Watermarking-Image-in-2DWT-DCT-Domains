import numpy as np
import pywt
import os
import cv2
from utils.zig_zag import *
from utils.dct import *
from utils.dwt import *
from utils.performance_metrices import *
from utils.filtering_attacks import *
from utils.geometrical_attacks import *
from utils.image_enhancement_attacks import *
from utils.noise_attacks import *
from utils.jpeg_compression import *
from PIL import Image
from scipy.ndimage import convolve

##############################################################################################################

alpha=1
current_path = str(os.path.dirname(__file__))

image = '5.2.08.tiff'   
watermark = 'Untitled.png' 

###############################################################################################################

def getGrayImage(img_name, size):
    # Open and resize the image
    img = cv2.imread('./dataset/' + img_name)
    img = cv2.resize(img, (size,size), interpolation=cv2.INTER_LINEAR)
    # Convert to grayscale
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb_img)

    return Y.astype(np.float64), Cr, Cb


##############################################################################################################

def embedWatermark(W, Y):
    W_flat = W.ravel()  # Flatten the watermark array
    ind = 0

    # Iterate over the image in 8x8 blocks
    for x in range(0, len(Y), 8):
        for y in range(0, len(Y), 8):
            if ind < len(W_flat):
                subdct = Y[x:x+8, y:y+8]
                subdct[5][5] = alpha*W_flat[ind]  # Embed watermark at [5][5] in each block
                Y[x:x+8, y:y+8] = subdct
                ind += 1

    return Y

##############################################################################################################

def blockDct(Y):
    size = len(Y)
    Y_dct = np.empty((size, size))
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            subpixels = Y[i:i+8, j:j+8]
            # Apply 2D DCT
            subdct = dct(dct(subpixels.T).T)  
            Y_dct[i:i+8, j:j+8] = subdct
    return Y_dct

###############################################################################################################

def invDct(Y_dct):
    size = len(Y_dct)
    Y_idct = np.empty((size, size))
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            # Apply 2D IDCT
            subidct = idct(idct(Y_dct[i:i+8, j:j+8].T).T)  
            Y_idct[i:i+8, j:j+8] = subidct
    return Y_idct

##############################################################################################################

def decodeWatermark(Y_ll1, z):
    subwatermarks = []
    for x in range(0, len(Y_ll1), 8):
        for y in range(0, len(Y_ll1), 8):
            coeff_slice = Y_ll1[x:x+8, y:y+8]
            subwatermarks.append(coeff_slice[5][5])  # Extract the embedded value from [5][5]
    W = np.array(subwatermarks).reshape(z, z)
    return W

#############################################################################################################

def getWatermark(Y_w, level=1):
    Y_w_dwt = dwt(Y_w,level)
    Y_w_dct = blockDct(Y_w_dwt[0:1024,0:1024])
    W = decodeWatermark(Y_w_dct, 128)
    print(np.min(W))
    print(np.max(W))
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i,j]< -50 or W[i,j]> 50:
                W[i,j]=255
            else:
                W[i,j]=0
    
    W = np.uint8(W)

    # Save the recovered watermark
    return W

###############################################################################################################

def print_image_from_array(Y, img_name,base_name):

    # Define the directory path
    directory_path = './dataset/' + base_name
    os.makedirs(directory_path, exist_ok=True)  # Create the directory if it doesn't exist

    # Define the full file path with _gray added to the name
    file_path = os.path.join(directory_path, img_name + '.jpg')

    # Save the image
    cv2.imwrite(file_path, Y)

#############################################################################################################

def w2d(img):
    model = 'haar'
    level = 1
    size = 2048
    z = 128

    # Convert and preprocess images
    Y, Cr, Cb = getGrayImage(image, size) # Y, W -> floating arrays 
    W, Cr_w, Cb_w = getGrayImage(watermark, z)
    # Process image coefficients
    Y_dwt = dwt(Y, 1)
    Y_dct = blockDct(Y_dwt[:size//2**level,:size//2**level]) # block dct of ll1

    # Embed the watermark into the DCT coefficients
    Y_w_dct = embedWatermark(W, Y_dct)
    Y_w_idct = invDct(Y_dct)

    # Reconstruct the watermarked image
    Y_dwt[:size//2**level,:size//2**level] = Y_w_idct # block dct of ll1

    Y_w = idwt(Y_dwt, level)

    #######################################################################

    Y_atk=jpeg_compression(Y_w,20)
    W_dec=getWatermark(Y_atk,1)
    base_name = os.path.splitext(image)[0]
    #orignal image
    print_image_from_array(Y,image,base_name)
    #watermarked image
    print_image_from_array(Y_w,image+'_watermark',base_name)
    #watermark
    print_image_from_array(W,'watermark',base_name)
    #attacked image
    print_image_from_array(Y_atk,'attacked_image',base_name)
    #decoded watermark after attack
    print_image_from_array(medianFilter(W_dec),'attacked_watermark',base_name)
    


# Run the watermark embedding and recovery
w2d("test")
