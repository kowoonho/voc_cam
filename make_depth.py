import os
import pathlib
import argparse
import imageio
import cv2
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
from voc12.dataloader import load_img_id_list
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = os.path.join(ROOT, "Dataset/VOC2012")

def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--depth_stage", default=10, type=int)
    parser.add_argument("--data_root", default=DATA, type=str)
    parser.add_argument("--train_list", default="voc12/trainval.txt", type=str)
    
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--model_type", default="DPT_Hybrid", type=str)
    
    parser.add_argument("--save_path", default="../result/depth_img", type=str)

    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    
    return args
def depth_normalize(depth_img):
    min_val = np.min(depth_img)
    max_val = np.max(depth_img)

    # 0에서 1 사이로 정규화
    depth_img_normalized = (depth_img - min_val) / (max_val - min_val)

    # 0에서 255 사이의 정수로 스케일링
    depth_img_int = (depth_img_normalized * 255).astype(np.uint8)
    
    return depth_img_int


def mean_depth(seg_img, depth_img):
    unique_values = np.unique(seg_img)
    
    mean_depth_img = np.zeros(seg_img.shape, dtype=np.float32)
    for values in unique_values:
        if values == 0: continue
        
        index = np.where(seg_img == values)
        
        mean_depth_img[index] = np.mean(depth_img[index], dtype=np.float32)
        
    return mean_depth_img

if __name__ == '__main__':
    args = get_arguments()
    
    img_path = os.path.join(args.data_root, "JPEGImages")
    seg_path = os.path.join(args.data_root, "SegmentationClass")
    
    img_id_list = load_img_id_list(args.train_list)
    
    model = torch.hub.load("intel-isl/MiDaS", args.model_type)
    model = model.to(args.device)
    model.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    
    if args.model_type == "DPT_Large" or args.model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    
    with torch.no_grad():
        for idx in tqdm(range(len(img_id_list))):
        # for idx in range(100):    
            img_id = img_id_list[idx]
            file_name = os.path.join(img_path, img_id + ".jpg")
            
            img = np.asarray(Image.open(file_name).convert("RGB"))
            transform_input = transform(img).to(args.device)
            
            prediction = model(transform_input)
            
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            depth_img = prediction.cpu().numpy()

            depth_img = depth_normalize(depth_img)
            
            save_name = os.path.join(args.save_path, img_id + ".png")
            imageio.imwrite(save_name, depth_img)
            # cv2.imwrite(save_name, depth_img)
            

        
        
    
    
    
    
    