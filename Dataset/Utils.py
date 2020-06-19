import numpy as np 
import cv2
import random


def Rotate(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def Flip(img, mode = 'v'):
    if mode == 'v':
        img = img[::-1, :,:]
    elif mode == 'h':
        img = img[:,::-1,:]
    else:
        raise Exception('mode = h or v')
    return img

def Saturation(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    adjust = random.uniform(0.5, 1.5)
    s = s * adjust
    s = np.clip(s, 0, 255).astype(hsv.dtype)
    hsv = cv2.merge((h, s, v))
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return bgr

def Hue(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    adjust = random.uniform(0.8, 1.2)
    h = h * adjust
    h = np.clip(h, 0, 255).astype(hsv.dtype)
    hsv = cv2.merge((h, s, v))
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return bgr

def Blur(bgr):
    ksize = random.choice([2, 3, 4, 5])
    bgr = cv2.blur(bgr, (ksize, ksize))
    return bgr

def Brightness(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    adjust = random.uniform(0.8, 1.2)
    v = v * adjust
    v = np.clip(v, 0, 255).astype(hsv.dtype)
    hsv = cv2.merge((h, s, v))
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return bgr

def Scale(img, scale):
    h, w, _ = img.shape
    img = cv2.resize(img, dsize=(
        int(w * scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    return img


def Gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def Crop(img, ratio):
    ratio = 0.45 if ratio > 0.5 or ratio <0 else ratio
    h1 = int(img.shape[0]*ratio)
    h2 = h1 + int((1 - ratio)*img.shape[0])
    w1 = int(img.shape[1]*ratio)
    w2 = w1 + int((1 - ratio)*img.shape[1])
    nimg = img.copy()[h1:h2, w1:w2]
    
    return nimg    
    
def rand(v = 0.6):
    if random.random() > v:
        return True
    return False

def CenterCrop(img, size):
    h, w,_ = img.shape
    xy1 = (w - size[0]) // 2
    xy2 = (h - size[1]) // 2
    return img[xy2:(xy2 + size[0]), xy1:(xy1 + size[1])]
    