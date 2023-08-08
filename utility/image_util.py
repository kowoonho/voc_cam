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
    
    if crop_box[2] > org_size[1]:
        crop_box = (crop_box[0], crop_box[1], int(org_size[1].item()), crop_box[3])
    if crop_box[3] > org_size[0]:
        crop_box = (crop_box[0], crop_box[1], crop_box[2], int(org_size[0].item()))
    
    box_size = (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])

    org_cam = np.zeros((org_size[0], org_size[1]), dtype=crop_cam.dtype)
    crop_cam = resize_with_interpolation(crop_cam, box_size)
    
    
    
    org_cam[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]] = crop_cam
    
    return org_cam

def resize_bbox(bbox, original_img_size, new_img_size):
    x1, y1, x2, y2 = bbox
    original_width, original_height = original_img_size
    new_width, new_height = new_img_size

    # Calculate the ratio of old size to new size
    width_ratio = new_width / original_width
    height_ratio = new_height / original_height

    # Resize the bounding box coordinates
    x1_new = int(x1 * width_ratio)
    y1_new = int(y1 * height_ratio)
    x2_new = int(x2 * width_ratio)
    y2_new = int(y2 * height_ratio)

    return (x1_new, y1_new, x2_new, y2_new)

def resize_bbox_list(bbox_list, original_img_size, new_img_size):
    new_bbox_list = []
    for bbox in bbox_list:
        new_bbox = resize_bbox(bbox, original_img_size, new_img_size)
        new_bbox_list.append(new_bbox)
        
    return new_bbox_list

def resize_with_interpolation(image, new_size):
    width, height = new_size
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized_image

def generate_random_color():
    # Generate random values for each RGB component
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)

    # Create a NumPy array with the RGB values
    color = np.array([red, green, blue], dtype=np.uint8)

    return color
    
def random_color_mask(mask):
    (w, h) = mask.shape
    gen_mask = np.zeros((w, h, 3), dtype=np.uint8)
    
    unique_values = np.unique(mask)
    
    for value in unique_values:
        if value == 0: continue
        random_color = generate_random_color()
        index = np.where(mask == value)
        gen_mask[index] = random_color
    
    return gen_mask

def crop_image(image): # numpy image cropping
    w, h = image.shape[:2]
    
    crop_images = []
    
    w_stride = w // 2
    h_stride = h // 2
    
    w_idx = [0, w_stride, w_stride*2]
    h_idx = [0, h_stride, h_stride*2]
    
    for i in range(2):
        for j in range(2):
            crop_images.append(image[w_idx[i]:w_idx[i+1], h_idx[j]:h_idx[j+1]])

    center_image = image[w//4:w//4 + w//2, h//4:h//4 + h//2]
    crop_images.append(center_image)
    
    return crop_images    

def merge_images(images, org_size):
    w, h = tuple(t.item() for t in org_size)
    
    
    if images[0].shape[0] < 20: # (C, H, W) => (H, W, C)
        for i in range(len(images)):
            images[i] = CHW_to_HWC(images[i])
    
    crop_w, crop_h = images[0].shape[:2]
            
    top = np.hstack((images[0], images[1]))
    bottom = np.hstack((images[2], images[3]))
    merge_image = np.vstack((top, bottom))
    
    merge_image = cv2.resize(merge_image, (h, w), interpolation=cv2.INTER_LINEAR)
    if len(merge_image.shape) == 2:
        merge_image = merge_image[:, :, np.newaxis]
    
    center_idx = [w//4, h//4, w//4 + crop_w, h//4 + crop_h]

    merge_image[center_idx[0]:center_idx[2], center_idx[1]:center_idx[3]] = (merge_image[center_idx[0]:center_idx[2], center_idx[1]:center_idx[3]] + images[4]) / 2
    
    final_image = HWC_to_CHW(merge_image)
    
    
    return final_image
    
    
    
    
    
    
    