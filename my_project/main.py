# my_project/main.py

import argparse
import os
import cv2, numpy as np
from utils.dwt import *
from utils.zig_zag import *
from utils.dct import *

def main():
    # Set up argument parsing for a single argument (image path)
    """parser = argparse.ArgumentParser(description="Command-line tool for image processing.")
    parser.add_argument("image_path", type=str, help="Path to the image file")

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
        raise ValueError("The provided file could not be read as an image.")"""

    print("Image loaded successfully. Proceeding with processing...")
    image = cv2.imread('example-images//vishal.jpg')
    image = cv2.resize(image, (512,512), interpolation=cv2.INTER_LINEAR)
    print("size of image is : ", image.shape)
    # ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # Y, Cr, Cb = cv2.split(ycrcb_img)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Y = gray_image
    cv2.imshow("watermarked_imgads.jpeg", Y)
    dwt_2l = dwt(Y, 2) 
    cv2.imshow("watermarked_imgdwt.jpeg", dwt_2l)
    m,n = Y.shape
    print("value of m, n is: ", m, n)
    # ll2 = dwt_2l[0:m//4, 0:n//4]
    # ll2_and_zig_zag = zig_zag(ll2)
    # print("size of zig-zag outpute is : ", ll2_and_zig_zag.shape)
    # v1 = ll2_and_zig_zag[0::2]
    # v2 = ll2_and_zig_zag[1::2]
    # dct_v1 = dct(v1)
    # dct_v2 = dct(v2)
    # # alpha can be wary
    # alpha = 0
    # z = dct_v1.shape[0] # watermark is added in v1 and v2 at a time
    # W = np.random.choice([0,1], size=z)
    # print("Random watermark is: ", W)
    # v1_w = 0.5*(dct_v1 + dct_v2) + alpha*W
    # v2_w = 0.5*(dct_v1 + dct_v2) - alpha*W
    # idct_v1 = idct(v1_w)
    # idct_v2 = idct(v2_w)

    # ll2_and_zig_zag_new = np.zeros(2*z)
    # for i in range(z):
    #     if(z%2):
    #         ll2_and_zig_zag_new[i] = idct_v2[i//2]
    #     else:
    #         ll2_and_zig_zag_new[i] = idct_v1[i//2]
    # print(ll2_and_zig_zag_new.shape)
    # ll2_new = zag_zig(ll2_and_zig_zag_new, m//4, n//4)
    # dwt_2l[0:m//4, 0:n//4] = ll2
    Y_new = idwt(dwt_2l, 2)

    # image_new = cv2.merge([dwt_2l_new.astype(np.uint8), Cr, Cb])
    cv2.imshow("watermarked_img.jpeg", Y_new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ensure main() runs when this script is called directly
if __name__ == "__main__":
    main()
