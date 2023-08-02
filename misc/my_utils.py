import os
import numpy as np
import torch
import cv2


def find_center_points(image_array):
    nonzero_indices = np.nonzero(image_array)
    points = np.column_stack((nonzero_indices[1], nonzero_indices[0]))
    center_points = np.median(points, axis = 0)
    return center_points

def find_threshold(image_array, type = "average", alpha = 1):
    
    image_array = np.where(image_array < 0.1, 0, image_array)
    
    nonzero_indices = np.nonzero(image_array)
    
    if type == "average":
        threshold = np.average(image_array[nonzero_indices])
    if type == "median":
        threshold = np.median(image_array[nonzero_indices])
    
    threshold *= alpha
    return threshold

def find_conf_cam(cam):
    threshold = find_threshold(cam)
    conf_cam = np.where(cam < threshold, 0, cam)
    
    return conf_cam

def find_crop_box(image_array, margin = 0.3):
    size = image_array.shape[:2]
    
    nonzero_indices = np.nonzero(image_array)
    
    min_x = np.min(nonzero_indices[1])
    max_x = np.max(nonzero_indices[1])
    min_y = np.min(nonzero_indices[0])
    max_y = np.max(nonzero_indices[0])
    
    bounding_box = (min_x, min_y, max_x, max_y)
    
    dx = max_x - min_x; dy = max_y - min_y;
    margin_dx = int(dx * margin); margin_dy = int(dy * margin);
    
    margin_dx = max(margin_dx, 15)
    margin_dy = max(margin_dy, 15)
    
    min_x = max(0, min_x - margin_dx)
    min_y = max(0, min_y - margin_dy)
    max_x = min(size[1], max_x + margin_dx)
    max_y = min(size[0], max_y + margin_dy)
    
    crop_box = (min_x, min_y, max_x, max_y)
    
    return crop_box

def visualize_bounding_box(image, bounding_box):
    x_min, y_min, x_max, y_max = bounding_box
    
    image_with_box = np.copy(image)
    if len(image_with_box.shape) != 3:
        image_with_box = cv2.cvtColor(image_with_box, cv2.COLOR_GRAY2RGB)
    
    cv2.rectangle(image_with_box, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return image_with_box

def crop_image_with_bounding_box(image, bounding_box):
    x_min, y_min, x_max, y_max = bounding_box
    cropped_image = image[y_min:y_max, x_min:x_max, :]
    return cropped_image

def visualize_total_box(image, cams):
    for cam in cams:
        conf_cam = find_conf_cam(cam)
        crop_box = find_crop_box(conf_cam)
        image = visualize_bounding_box(image, crop_box)
        
    return image
