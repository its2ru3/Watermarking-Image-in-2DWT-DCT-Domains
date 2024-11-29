import cv2
import numpy as np
import pywt
from scipy.fftpack import dct, idct
import random
import os

# Read the image
image_path = 'example-images\\4.1.03.tiff'  # Directly set the image path here
img = cv2.imread(image_path)

# Convert to grayscale and double precision
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('original_img.jpg', img_gray)
img_double = np.double(img_gray) / 255.0

A = img_double  # Host image

# DWT Decomposition
LLr, (LHr, HLr, HHr) = pywt.dwt2(A, 'db1')
LLr2, (LHr2, HLr2, HHr2) = pywt.dwt2(LLr, 'db1')

I = LLr2
sizeofllr = I.shape  # Subband size

b = 255  # Length of watermark
wp = np.random.randint(-2, 2, b)

w = np.where(wp > -1, 1, -1)
wt = I

# Random location selection
n = 1
m = np.arange(2, b+2)
Top = np.zeros((n, len(m)))
for o in range(n):
    Top[o, :] = np.random.permutation(m)

number = Top
numbernew = Top
r, c = wt.shape
n = r

# Zigzag
zig = []
k = 0
for i in range(1, n+1):
    for j in range(1, i+1):
        if i % 2 == 0:
            zig.append(wt[j-1, i-j])
        else:
            zig.append(wt[i-j, j-1])
        k += 1

for i in range(1, n):
    for j in range(1, n-i+1):
        if i % 2 == 0:
            zig.append(wt[n-j, j+i-1])
        else:
            zig.append(wt[i+j-1, n-j])

zig = np.array(zig)

# Create parallel vectors
x1 = zig[1::2]
x2 = zig[::2]

# DCT
x1d = dct(x1, norm='ortho')
x2d = dct(x2, norm='ortho')

# Embedding
number_flat = number.flatten().astype(int)  # Flatten the array to make it 1D
xp1 = 0.5 * (x1d[number_flat] + x2d[number_flat]) + 0.1 * w
xp2 = 0.5 * (x1d[number_flat] + x2d[number_flat]) - 0.1 * w

x1d[number_flat] = xp1
x2d[number_flat] = xp2

# Inverse DCT
xid1 = idct(x1d, norm='ortho')
xid2 = idct(x2d, norm='ortho')

x = np.zeros_like(zig)
x[1::2] = xid1
x[::2] = xid2

# Inverse Zigzag
inz = np.zeros_like(wt)
k = 0
for i in range(1, n+1):
    for j in range(1, i+1):
        if i % 2 == 0:
            inz[j-1, i-j] = x[k]
        else:
            inz[i-j, j-1] = x[k]
        k += 1

for i in range(1, n):
    for j in range(1, n-i+1):
        if i % 2 == 0:
            inz[n-j, j+i-1] = x[k]
        else:
            inz[i+j-1, n-j] = x[k]
        k += 1

# Inverse DWT
preoriginalX = pywt.idwt2((inz, (LHr2, HLr2, HHr2)), 'db1')
originalX = pywt.idwt2((preoriginalX, (LHr, HLr, HHr)), 'db1')

# PSNR Function
def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

psnr_value = psnr(originalX, A)
print(f"PSNR value: {psnr_value}")

# Save the watermarked image
output_image = np.uint8(originalX * 255)

cv2.imwrite('pexfn2.jpg', output_image)

# Image attack test (example with compression)
cv2.imwrite('pexfn2_compressed.jpg', output_image, [cv2.IMWRITE_JPEG_QUALITY, 50])
