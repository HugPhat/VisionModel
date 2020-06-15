import numpy as np 
import cv2
import random

def SegCrop(img, mask, value):
    pass

def SegRotate(img, value):
    pass

def SegFlip(img, mode = 'v'):
    if mode == 'v':
        img = img[::-1, :,:]
    elif mode == 'h':
        img = img[:,::-1,:]
    else:
        raise Exception('mode = h or v')
    return img

def SegSaturation(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    adjust = random.uniform(0.5, 1.5)
    s = s * adjust
    s = np.clip(s, 0, 255).astype(hsv.dtype)
    hsv = cv2.merge((h, s, v))
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return bgr

def SegHue(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    adjust = random.uniform(0.8, 1.2)
    h = h * adjust
    h = np.clip(h, 0, 255).astype(hsv.dtype)
    hsv = cv2.merge((h, s, v))
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def SegBlur(bgr):
    ksize = random.choice([2, 3, 4, 5])
    bgr = cv2.blur(bgr, (ksize, ksize))
    return bgr

def SegBrightness(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    adjust = random.uniform(0.8, 1.2)
    v = v * adjust
    v = np.clip(v, 0, 255).astype(hsv.dtype)
    hsv = cv2.merge((h, s, v))
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def SegScale(img, scale):
    h, w, _ = img.shape
    img = cv2.resize(img, dsize=(
        int(w * scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    return img

def rand():
    if random.random() > 0.5:
        return True
    return False