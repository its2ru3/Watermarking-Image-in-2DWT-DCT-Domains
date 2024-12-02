import cv2, numpy as np
def jpeg_compression(Y_new, compression_factor): # using inbuilt function(imencode and imdecode) to compress the original image with the quality factor from user
    _, compressed_image = cv2.imencode('.jpg', Y_new, [int(cv2.IMWRITE_JPEG_QUALITY), 100-compression_factor])
    decompressed_image = cv2.imdecode(compressed_image, cv2.IMREAD_UNCHANGED)
    return np.array(decompressed_image)