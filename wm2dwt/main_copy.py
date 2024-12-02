# my_project/main.py
import argparse
import os, csv
import cv2, numpy as np
import pywt
from scipy.fft import dct as DCT, idct as iDCT
from utils.zig_zag import *
from utils.dct import *
from utils.dwt import *
from utils.performance_metrices import *
from utils.filtering_attacks import *
from utils.geometrical_attacks import *
from utils.image_enhancement_attacks import *
from utils.noise_attacks import *
from utils.jpeg_compression import *

def encode(Y, alpha, len_w, L=2):
    m,n = Y.shape
    img_double = np.double(Y) / 255.0
    A = img_double  # Host image
    print("size of img_double: ", A.shape)
    # DWT Decomposition
    LLr, (LHr, HLr, HHr) = pywt.dwt2(A, 'db1')
    LLr2, (LHr2, HLr2, HHr2) = pywt.dwt2(LLr, 'db1')

    dwt_2l = LLr2
    ll2 = dwt_2l[0:m//2**L, 0:n//2**L]
    ll2_and_zig_zag = zig_zag(ll2)

    v1 = ll2_and_zig_zag[0::2]
    print("v1 for the image is : \n", v1)
    v2 = ll2_and_zig_zag[1::2]
    dct_v1 = DCT(v1, type=2, norm='ortho')
    print("DCT coeffecient of v1 is: \n", dct_v1)
    dct_v2 = DCT(v2, type=2, norm='ortho')
    z = dct_v1.shape[0] 
    W = np.random.choice([1,-1], size=len_w)
    v1_w=np.copy(dct_v1)
    v2_w=np.copy(dct_v2)
    #here
    v1_w[-len_w:] = 0.5*(dct_v1[-len_w:] + dct_v2[-len_w:]) + alpha*W
    v2_w[-len_w:] = 0.5*(dct_v1[-len_w:] + dct_v2[-len_w:]) - alpha*W
    print("v1_w is \n", v1_w)

    # v1_w[-len_w:] = 0.5*(dct_v1[-len_w:] + dct_v2[-len_w:]) + alpha*W
    # v2_w[-len_w:] = 0.5*(dct_v1[-len_w:] + dct_v2[-len_w:]) - alpha*W

    idct_v1 = iDCT(v1_w, type=2, norm='ortho')
    print("idct_v1 is \n", idct_v1)
    idct_v2 = iDCT(v2_w, type=2, norm='ortho')
    ll2_and_zig_zag_new = np.zeros(2*z)
    ll2_and_zig_zag_new[0::2]=idct_v1
    ll2_and_zig_zag_new[1::2]=idct_v2

    ll2_new = zag_zig(ll2_and_zig_zag_new, m//2**L, n//2**L)
    inz = ll2_new    

    preoriginalX = pywt.idwt2((inz, (LHr2, HLr2, HHr2)), 'db1')
    Y_new = pywt.idwt2((preoriginalX, (LHr, HLr, HHr)), 'db1')
    return Y_new, W

def decode(Y_new, len_w, L=2):
    m,n = Y_new.shape
    img_double = np.double(Y) / 255.0
    A = img_double  # Host image
    print("size of img_double: ", A.shape)
    # DWT Decomposition
    LLr, (LHr, HLr, HHr) = pywt.dwt2(A, 'db1')
    LLr2, (LHr2, HLr2, HHr2) = pywt.dwt2(LLr, 'db1')

    dwt_2l = LLr2

    ll2 = dwt_2l[0:m//2**L, 0:n//2**L]
    ll2_and_zig_zag = zig_zag(ll2)

    v1 = ll2_and_zig_zag[0::2]
    v2 = ll2_and_zig_zag[1::2]
    dct_v1 = DCT(v1, type=2, norm='ortho')
    dct_v2 = DCT(v2, type=2, norm='ortho')
    #here
    W_=dct_v1[-len_w:] - dct_v2[-len_w:]
    # print("watermark unnormalised is: \n", W_)
    W_dec = np.where(W_ <= 0, -1, np.where(W_ > 0, 1, 0))
    return W_dec

def attacks(Y_new, len_w, W, atk_type):
    print("\n Gamma Correction")
    Y_atk_gamma_corr = gammaCorrection(Y_new, 0.5)
    print("datatype of gamma correction image is: ", Y_atk_gamma_corr.dtype)
    # cv2.imshow("Median_atk.jpg", Y_atk_median)
    W_dec = decode(Y_atk_gamma_corr, len_w)
    print("Watermark from attacked image: \n", W_dec)
    print("BCR of attacked image is: ", bcr(W, W_dec))
    print("PSNR of attacked image is: ", psnr(Y_new, Y_atk_gamma_corr))
    print("SSIM of attacked image is: ", ssim(Y_new, Y_atk_gamma_corr))

    print("\n Gaussian noise ")
    Y_atk_gaussian = gaussian_noise_adder(Y_new, 0.0005)
    W_dec = decode(Y_atk_gaussian, len_w)
    print("Watermark from attacked image: \n", W_dec)
    print("BCR of attacked image is: ", bcr(W, W_dec))
    print("PSNR of attacked image is: ", psnr(Y_new, Y_atk_gaussian))
    print("SSIM of attacked image is: ", ssim(Y_new, Y_atk_gaussian))


    

def wm2dwt():
    # Set up argument parsing for a single argument (image path)
    parser = argparse.ArgumentParser(description="Command-line tool for watermarking an image using 2DWT.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("-a", type=str, default=None, metavar="atk", help="Type of attack to check on the image")
    parser.add_argument("-l", type=int, default=2, metavar="L", help="Level of DWT to use for embedding the watermark")
    parser.add_argument("--alpha", type=float, default=0.1, metavar="alpha", help="Gain index for the watermark")
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
    cv2.imwrite("orginal_colored_image.jpeg", image)
    print("size of image is : ", image.shape)
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb_img)
    # cv2.imshow("gray_image_wm.jpeg", Y)
    Y = Y.astype(np.float64)

    csv_file_path = 'original_image.csv'
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(Y)
    print("datatype of original image is: ", Y.dtype)

    alpha = args.alpha
    len_w = args.len_w
    L = args.l
    Y_new, W_enc = encode(Y, alpha, len_w)
    # print("datatype of Y_new image is: ", Y_new.dtype)
    print("Random watermark is:\n", W_enc)

    cv2.imwrite("watermarked_img.jpeg", cv2.cvtColor(cv2.merge([Y_new.astype(np.uint8), Cr, Cb]), cv2.COLOR_YCrCb2BGR))
    W_dec = decode(Y_new, len_w)
    print("Decoded watermark is:\n", W_dec)
    print("bcr is ", bcr(W_enc, W_dec))
    print("psnr to Y is ", psnr(Y, Y_new))

    atk_type = "filtering_attacks"
    # attacks(Y_new, len_w, W_enc, atk_type)

    cv2.waitKey(0)
    cv2.destroyAllWindows
    
# Ensure wm2dwt() runs when this script is called directly
if __name__ == "__main__":
    try:
        wm2dwt()
    except ValueError as e:
        print(e)