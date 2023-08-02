import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
import cv2
from .util import *
import random

def read_image(filename, mode = 'L'):
    image = Image.open(filename)
    
    if mode == "L":
        img = np.array(image)
        
    elif mode == "RGB":
        img = np.array(image.convert('RGB'))
        
    else:
        raise ValueError("mode name is not proper")

    return img

def image_show(*files, mode='L', idx=(1, 1)):
    fig = plt.figure()
    ax = []
    
    for i, file in enumerate(files):
        if isinstance(file, np.ndarray) == False:
            img = read_image(file, mode)
        else:
            img = file
        ax.append(fig.add_subplot(idx[0], idx[1], i + 1))
        ax[i].imshow(img)
        ax[i].axis('off')
        
    plt.show()


def read_label(file, dtype=np.int32):
    f = Image.open(file)
    try:
        img = f.convert('P')
        img = np.array(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        return img
    elif img.shape[2] == 1:
        return img[:, :, 0]
    else:
        raise ValueError("Color image can't be accepted as label image.")

def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))

def CHW_to_HWC(img):
    return np.transpose(img, (1, 2, 0))




        

def scaling(img, factor = 1.):
    if img.ndim == 2:
        (h, w) = img.shape
        scale_image = cv2.resize(img, (int(w * factor), int(h * factor)))
          
    elif img.ndim == 3:
        if img.shape[0] == 3: # (3, H, W)
            (h, w) = img.shape[1:3]
            img = CHW_to_HWC(img)
            scale_image = cv2.resize(img, (int(w * factor), int(h * factor)))
            scale_image = HWC_to_CHW(scale_image)
            
        elif img.shape[2] == 3: # (H, W, 3)
            (h, w) = img.shape[:2]
            scale_image = cv2.resize(img, (int(w * factor), int(h * factor)))
            
        else:
            raise ValueError("Image shape is not proper.")
    else:
        raise ValueError("Image shape is not proper.")
    
    return scale_image

def img_resize(img, rescale_size):
    if img.ndim == 2:
        scale_image = cv2.resize(img, (rescale_size[1], rescale_size[0]))
          
    elif img.ndim == 3:
        if img.shape[0] == 3: # (3, H, W)

            img = CHW_to_HWC(img)
            scale_image = cv2.resize(img, (rescale_size[1], rescale_size[0]))
            scale_image = HWC_to_CHW(scale_image)
            
        elif img.shape[2] == 3: # (H, W, 3)
            scale_image = cv2.resize(img, (rescale_size[1], rescale_size[0]))
            
        else:
            raise ValueError("Image shape is not proper.")
    else:
        raise ValueError("Image shape is not proper.")
    
    return scale_image

       
    
def return_org_img(crop_image, bbox, org_size):
    crop_size = (int(bbox[3]-bbox[1]), int(bbox[2]-bbox[0]))
    
    crop_image = img_resize(crop_image, crop_size)
    
    img = np.zeros((org_size[0], org_size[1]), dtype=crop_image.dtype)
    img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = crop_image
    
    return img

def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw


def random_crop(images, cropsize, default_values):

    if isinstance(images, np.ndarray): images = (images,)
    if isinstance(default_values, int): default_values = (default_values,)

    imgsize = images[0].shape[:2]
    box = get_random_crop_box(imgsize, cropsize)

    new_images = []
    for img, f in zip(images, default_values):

        if len(img.shape) == 3:
            cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
        else:
            cont = np.ones((cropsize, cropsize), img.dtype)*f
        cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
        new_images.append(cont)

    if len(new_images) == 1:
        new_images = new_images[0]

    return new_images

def crop_cam_to_org_cam(crop_cam, crop_box, org_size):
    
    org_cam = np.zeros((org_size[0], org_size[1]), dtype=crop_cam.dtype)
    
    org_cam[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]] = crop_cam
    
    return org_cam
    