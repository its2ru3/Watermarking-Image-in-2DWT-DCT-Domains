import numpy as np

# check function for boundries
def c(i,j,n,m):
    if not(i<n and i>=0):
        return False
    if not(j<m and j>=0):
        return False
    return True


# image to zig zag
def zig_zag(img) -> np.ndarray:
    n,m=img.shape
    l=[]
    row = 0
    col = 0
    rev=True
    while row < n:
        i = row
        j = col
        temp = []
        while c(i, j, n, m):
            temp.append(img[i, j])
            i += 1
            j -= 1
        if temp:
            if rev:
                l=l+temp[::-1]
                rev=False
            else:
                l=l+temp
                rev=True
        if col < m - 1:
            col += 1
        else:
            row += 1

    return np.array(l)


# zig zag to image
def zag_zig(l,n,m) -> np.ndarray:
    img=np.zeros((n,m))
    row=0
    col=0
    ind=0
    rev=True
    while row < n:
        i = row
        j = col
        if rev:
            rev = False
            stack=[]
            while c(i, j, n, m):
                stack.append([i,j])
                i += 1
                j -= 1
            while stack:
                img[stack[-1][0],stack[-1][1]]=l[ind]
                stack.pop()
                ind+=1    
        else:
            rev = True
            while c(i, j, n, m):
                img[i, j]=l[ind]
                ind+=1
                i += 1
                j -= 1
        if col < m - 1:
            col += 1
        else:
            row += 1
    return img


