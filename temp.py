# my_project/main.py

import argparse
import os
import cv2, numpy as np
from utils.dwt import *
from utils.zig_zag import *
from utils.dct import *
from utils.performance_metrices import *
from utils.image_enhancement_attacks import *

def encode(Y):
    m,n = Y.shape
    dwt_2l = dwt(Y, 2) 
    ll2 = dwt_2l[0:m//4, 0:n//4]
    ll2_and_zig_zag = zig_zag(ll2)

    v1 = ll2_and_zig_zag[0::2]
    v2 = ll2_and_zig_zag[1::2]
    dct_v1 = dct(v1)
    dct_v2 = dct(v2)
    alpha = 0.1
    z = dct_v1.shape[0] 
    len_w=128
    W = np.random.choice([1,-1], size=len_w)
    print("Random watermark is: ", W)
    v1_w=np.copy(dct_v1)
    v2_w=np.copy(dct_v2)
    v1_w[-len_w:] = 0.5*(dct_v1[-len_w:] + dct_v2[-len_w:]) + alpha*W
    # print(v1_w[-len_w:])
    v2_w[-len_w:] = 0.5*(dct_v1[-len_w:] + dct_v2[-len_w:]) - alpha*W
    # print("\n", v2_w[-len_w:])
    # print("\n",0.5*(dct_v1[-len_w:] + dct_v2[-len_w:]))
    idct_v1 = idct(v1_w)
    idct_v2 = idct(v2_w)
    ll2_and_zig_zag_new = np.zeros(2*z)
    ll2_and_zig_zag_new[0::2]=idct_v1
    ll2_and_zig_zag_new[1::2]=idct_v2

    ll2_new = zag_zig(ll2_and_zig_zag_new, m//4, n//4)
    dwt_2l[0:m//4, 0:n//4] = ll2_new
    Y_new = idwt(dwt_2l, 2)
    return Y_new, W

def decode(Y_new):
    m,n = Y_new.shape
    dwt_2l = dwt(Y_new, 2) 

    ll2 = dwt_2l[0:m//4, 0:n//4]
    ll2_and_zig_zag = zig_zag(ll2)

    v1 = ll2_and_zig_zag[0::2]
    v2 = ll2_and_zig_zag[1::2]
    dct_v1 = dct(v1)
    dct_v2 = dct(v2)
    z = dct_v1.shape[0] 
    len_w=128
    # v1_w=dct_v1
    # v2_w=dct_v2
    W_=dct_v1[-len_w:] - dct_v2[-len_w:]
    W_dec = np.where(W_ < 0, -1, np.where(W_ > 0, 1, 0))
    print("Random watermark is: ", W_dec)
    return W_dec

# def attacks(img):
#     Y = img
#     Y_color = cv2.merge((Y, Cr, Cb))
#     Y_new_color = cv2.merge((Y_new, Cr, Cb))
#     bitPlaneRemoved_img = bitPlaneRemoval(Y_new_color.astype(np.unit8),2)
#     bitPlaneRemoved_img_ = cv2.cvtColor(bitPlaneRemoved_img, cv2.COLOR_BGR2YCrCb)
#     bitPlaneRemoved_img_gray, Cr, Cb = cv2.split(bitPlaneRemoved_img_)
#     W_dec = decode(bitPlaneRemoved_img_gray)

def main_():
    print("Image loaded successfully. Proceeding with processing...")
    image = cv2.imread('example-images//4.2.07.tiff')
    image = cv2.resize(image, (512,512), interpolation=cv2.INTER_LINEAR)
    print("size of image is : ", image.shape)
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb_img)
    Y = Y.astype(np.float64)
    cv2.imshow("img.jpeg", Y.astype(np.uint8))
    Y_new, W_enc = encode(Y)
    cv2.imshow("watermarked_img.jpeg", Y_new.astype(np.uint8))
    W_dec = decode(Y_new)
    print("bcr is ", bcr(W_enc, W_dec))
   
    print("bcr is ", bcr(W_enc, W_dec))
    print("psnr to Y is ", psnr(Y_color, bitPlaneRemoved_img_gray))
    print("psnr to Y_new is ", psnr(Y_new_color, bitPlaneRemoved_img_gray))
    cv2.waitKey(0)
    cv2.destroyAllWindows
# Ensure main() runs when this script is called directly
# if _name_ == "_main_":
main_()