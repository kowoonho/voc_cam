import numpy as np
import cv2
def box_per_depth(depth_value, box_size):
    if depth_value < 5:
        factor = 1.0
        
    else:
        box_level = box_size // 40000
        if box_level == 0: box_level = 1
        
        factor = depth_value / (3 * box_level)
        
        if factor >= 2.5:
            factor = 2.5

    return round(factor, 1)
        
        

def bbox_size(bbox):
    if isinstance(bbox, list):
        bbox_sizes = []
        for box in bbox:
            bbox_sizes.append((box[2] - box[0]) * (box[3] - box[1]))
        return bbox_sizes
    else:
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
def box2shape(bbox):
    if isinstance(bbox, list):
        bbox_shapes = []
        for box in bbox:
            bbox_shapes.append((box[3] - box[1], box[2] - box[0]))
        return bbox_shapes
    else:
        return (bbox[3] - bbox[1], bbox[2] - bbox[0])
    
def find_center_points(image_array):
    nonzero_indices = np.nonzero(image_array)
    points = np.column_stack((nonzero_indices[1], nonzero_indices[0]))
    center_points = np.median(points, axis = 0)
    return center_points

def find_threshold(image_array, type = "average", alpha = 0.4):
    
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

def compute_iou(pred_mask, gt_mask): # (H, W)
        
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    area_pred_mask = pred_mask.sum()
    area_gt_mask = gt_mask.sum()
        
    union = area_pred_mask + area_gt_mask - intersection
        
    if union == 0:
        iou = 0
    else:
        iou = intersection / union
    
    return iou
    


def depth_iou(pred, gt, mean_depth, box_size):
    label = np.unique(gt)
    label = [x for x in label if x not in [-1, 0]]
    
    out = []
    
    # for idx, cls in enumerate(label):
    #     # if idx >= len(mean_depth) or idx >= len(box_size):
    #     #     print("label_length : {}".format(len(label)))
    #     #     print("mean_depth_length : {}".format(len(mean_depth)))
    #     #     print("box_size_length : {}".format(len(box_size)))
    #     #     raise IndexError("Index out of range for mean_depth or box_size.")
    #     if idx >= len(box_size):
    #         iou = 0.
    #         factor = 0.
    #     else:
    #         factor = mean_depth[idx] / (box_size[idx] // 10000)
    
    #         pred_mask = np.where(pred == cls, 1, 0)
    #         gt_mask = np.where(gt == cls, 1, 0)
            
    #         iou = compute_iou(pred_mask, gt_mask)
        
    #     out.append((round(factor, 3), round(iou, 3)))
        
    for idx in range(len(box_size)):
        if idx >= len(label): continue
        
        pred_mask = np.where(pred == label[idx], 1, 0)
        gt_mask = np.where(gt == label[idx], 1, 0)
    
        iou = compute_iou(pred_mask, gt_mask)
        
        out.append((round(box_size[idx] / 10000, 3), round(iou, 3)))
        
    return out

def refine_cam(cams, threshold=0.3):
    cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', 
                  constant_values=threshold)
    return np.argmax(cams, axis = 0)

def mask_image(image, mask_region): # mask_region = (H, W)
    expanded_mask = np.repeat(mask_region[:, :, np.newaxis], 3, axis=2)
    masked_image = image * expanded_mask
    
    return masked_image

def mean_depth(depth_map, gt, depth_stage = 10):
    label = np.unique(gt)
    
    mean_depth_map = np.zeros(gt.shape, dtype=np.int32)
    
    for value in label:
        if value == -1: continue
        value_idx = np.where(gt == value)
        mean_depth = np.average(depth_map[value_idx])
        
        mean_depth_map[value_idx] = depth_stage - round(mean_depth)
    
    mean_depth_map = mean_depth_map - 1
    return mean_depth_map

def mean_depth_value(depth_map, gt, depth_stage = 0, mode = "average"):
    label = np.unique(gt)
    
    mean_depth_list = []
    
    for value in label:
        if value == 0 or value == 255 or value == -1: continue
        indexes = np.where(gt == value)
        
        if mode == "average":
            mean_depth = np.average(depth_map[indexes])
        elif mode == "median":
            mean_depth = np.median(depth_map[indexes])
        else:
            raise ValueError("mode should be selected only average or median.")
        
        if depth_stage == 0:
            mean_depth_list.append(mean_depth)
        else:
            mean_depth_list.append(depth_stage - round(mean_depth) - 1)
            
        
    return np.array(mean_depth_list)

    

        

def get_pseudo_label(cam_dict):
    cams = np.pad(cam_dict['high_res'], ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.35)
    keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
    pseudo = np.argmax(cams, axis = 0)
    pseudo = keys[pseudo]
    
    return pseudo

def sort_cam(depth_map, cams):
    ref_cam = refine_cam(cams)
    
    mean_depth = mean_depth_value(depth_map, ref_cam, mode = 'median')
    
    sorted_idx = np.argsort(mean_depth)[::-1]
    
    return sorted_idx
    