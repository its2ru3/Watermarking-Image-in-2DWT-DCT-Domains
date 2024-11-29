# my_project/main.py
import argparse
import os
import cv2, numpy as np
from utils.zig_zag import *
from utils.dct import *
from utils.dwt import *
from utils.performance_metrices import *

def encode(Y, alpha, len_w, L):
    m,n = Y.shape
    dwt_2l = dwt(Y, L) 
    ll2 = dwt_2l[0:m//4, 0:n//4]
    ll2_and_zig_zag = zig_zag(ll2)

    v1 = ll2_and_zig_zag[0::2]
    v2 = ll2_and_zig_zag[1::2]
    dct_v1 = dct(v1)
    dct_v2 = dct(v2)
    z = dct_v1.shape[0] 
    W = np.random.choice([1,-1], size=len_w)
    v1_w=np.copy(dct_v1)
    v2_w=np.copy(dct_v2)
    v1_w[-len_w:] = 0.5*(dct_v1[-len_w:] + dct_v2[-len_w:]) + alpha*W
    v2_w[-len_w:] = 0.5*(dct_v1[-len_w:] + dct_v2[-len_w:]) - alpha*W
    idct_v1 = idct(v1_w)
    idct_v2 = idct(v2_w)
    ll2_and_zig_zag_new = np.zeros(2*z)
    ll2_and_zig_zag_new[0::2]=idct_v1
    ll2_and_zig_zag_new[1::2]=idct_v2

    ll2_new = zag_zig(ll2_and_zig_zag_new, m//4, n//4)
    dwt_2l[0:m//4, 0:n//4] = ll2_new
    Y_new = idwt(dwt_2l, L)
    return Y_new, W

def decode(Y_new, len_w, L):
    m,n = Y_new.shape
    dwt_2l = dwt(Y_new, L) 

    ll2 = dwt_2l[0:m//4, 0:n//4]
    ll2_and_zig_zag = zig_zag(ll2)

    v1 = ll2_and_zig_zag[0::2]
    v2 = ll2_and_zig_zag[1::2]
    dct_v1 = dct(v1)
    dct_v2 = dct(v2)

    W_=dct_v1[-len_w:] - dct_v2[-len_w:]
    W_dec = np.where(W_ < 0, -1, np.where(W_ > 0, 1, 0))
    return W_dec

# def attacks(img):
#     Y = img
#     Y_color = cv2.merge((Y, Cr, Cb))
#     Y_new_color = cv2.merge((Y_new, Cr, Cb))
#     bitPlaneRemoved_img = bitPlaneRemoval(Y_new_color.astype(np.unit8),2)
#     bitPlaneRemoved_img_ = cv2.cvtColor(bitPlaneRemoved_img, cv2.COLOR_BGR2YCrCb)
#     bitPlaneRemoved_img_gray, Cr, Cb = cv2.split(bitPlaneRemoved_img_)
#     W_dec = decode(bitPlaneRemoved_img_gray)

def wm2dwt():
    # Set up argument parsing for a single argument (image path)
    parser = argparse.ArgumentParser(description="Command-line tool for watermarking an image using 2DWT.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("-a", type=str, default=None, metavar="atk", help="Type of attack to check on the image")
    parser.add_argument("-l", type=int, default=2, metavar="L", help="Level of DWT to use for embedding the watermark")
    parser.add_argument("--alpha", type=int, default=0.1, metavar="alpha", help="Gain index for the watermark")
    parser.add_argument("-z","--len_w", type=int, default=128, metavar="len_w", help="length of watermark")

    try:
        args = parser.parse_args()
    except:
        # If there is any error, print the help message
        parser.print_help()
        return

    args = parser.parse_args()
    # Check if the file exists and has a valid image extension
    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"File '{args.image_path}' does not exist.")
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    _, ext = os.path.splitext(args.image_path)
    if ext.lower() not in valid_extensions:
        raise ValueError("The provided file is not a valid image format.")

    # Attempt to load the image to verify it’s a readable image file
    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError("The provided file could not be read as an image.")



    print("\nImage loaded successfully. Proceeding with processing...\n")
    print("size of original image is : ", image.shape)
    # ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # Y, Cr, Cb = cv2.split(ycrcb_img)
    image = cv2.resize(image, (512,512), interpolation=cv2.INTER_LINEAR)
    print("size of image is : ", image.shape)
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb_img)
    cv2.imshow("gray_image_wm.jpeg", Y)
    Y = Y.astype(np.float64)
<<<<<<< HEAD
    cv2.imshow("gray_image_wm.jpeg", Y.astype(np.uint8))
=======
>>>>>>> 425ae828eda4615d29b7e4ec0f6a8889c73bdcbf

    alpha = args.alpha
    len_w = args.len_w
    L = args.l
    Y_new, W_enc = encode(Y, alpha, len_w, L)
    print("Random watermark is: ", W_enc)

    cv2.imshow("watermarked_img.jpeg", Y_new.astype(np.uint8))
    W_dec = decode(Y_new, len_w, L)
    print("Decoded watermark is: ", W_dec)

    print("bcr is ", bcr(W_enc, W_dec))
   
    print("psnr to Y is ", psnr(Y, Y_new))
    cv2.waitKey(0)
    cv2.destroyAllWindows
    
# Ensure wm2dwt() runs when this script is called directly
if __name__ == "__main__":
    try:
        wm2dwt()
    except ValueError as e:
        print(e)