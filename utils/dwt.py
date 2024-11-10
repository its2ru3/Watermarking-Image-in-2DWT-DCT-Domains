import numpy as np

# convolution of img with HAAR scaling function and HAAR wavelet in vertical direction
def con_v(img, highpass=False):
    if highpass:
        return (img[:, 0::2] - img[:, 1::2]) / np.sqrt(2)
    else:
        return (img[:, 0::2] + img[:, 1::2]) / np.sqrt(2)


# convolution of img with HAAR scaling function and HAAR wavelet in horizontal direction
def con_h(img, highpass=False):
    if highpass:
        return (img[0::2] - img[1::2]) / np.sqrt(2)
    else:
        return (img[0::2] + img[1::2]) / np.sqrt(2)


# dwt of img at Level "level" using function con_v and con_h
def dwt(img:np.ndarray, level:int) -> np.ndarray:
    if level<=0:
        raise ValueError("Level value should be greater than 0.")
    n,m=img.shape
    r=np.zeros((n, m),dtype=np.float64)
    
    lf = con_v(con_h(img),True)
    fl = con_v(con_h(img,True))
    ff = con_v(con_h(img,True),True)
    
    if level == 1:
        ll = con_v(con_h(img))
    else:
        ll = dwt(con_v(con_h(img)), level -1)

    r[0:n//2, 0:m//2] = ll
    r[n//2:n, 0:m//2] = lf
    r[0:n//2, m//2:m] = fl
    r[n//2:n, m//2:m] = ff

    return r


# idwt of dwt of img
def idwt(img:np.ndarray, level:int) -> np.ndarray:
    if level<=0:
        raise ValueError("Level value should be greater than 0.")
    
    n, m = img.shape
    
    r1 = np.zeros((n//2,m),dtype=np.float64)
    r2 = np.zeros((n//2,m),dtype=np.float64)
    r = np.zeros((n,m),dtype=np.float64)
    
    lf = img[0:n//2, m//2:m]
    fl = img[n//2:n, 0:m//2]
    ff = img[n//2:n, m//2:m]
    
    if level == 1:
        ll = img[0:n//2, 0:m//2]
    else:
        ll = idwt(img[0:n//2, 0:m//2],level - 1)
        
    r1[:, 0::2] = (ll + fl) / np.sqrt(2)
    r1[:, 1::2] = (ll - fl) / np.sqrt(2)
    r2[:, 0::2] = (lf + ff) / np.sqrt(2)
    r2[:, 1::2] = (lf - ff) / np.sqrt(2)
  
    r[0::2, :] = (r1 + r2) / np.sqrt(2)
    r[1::2, :] = (r1 - r2) / np.sqrt(2)

    return r


