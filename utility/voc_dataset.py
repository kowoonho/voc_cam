import numpy as np
import os
from torch.utils.data import Dataset
from .image_util import *
from .util import *

class VOCSemanticSegmentationDataset(Dataset):
    def __init__(self, data_dir = "../Dataset/VOC2012", split = 'val', depth_dir = ""):
        if split not in ['train', 'trainval', 'val']:
            raise ValueError(
                'please pick split from \'train\', \'trainval\', \'val\'')
        
        # id_list_file = os.path.join(
        #     data_dir, 'ImageSets/Segmentation/{0}.txt'.format(split))
        id_list_file = os.path.join("voc12/{}.txt".format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir
        self.depth_dir = depth_dir
        self.cam_dir = "../irn_result/cam"

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        img_path = os.path.join(
            self.data_dir, 'JPEGImages', self.ids[i] + '.jpg')
        img = read_image(img_path, mode = 'L')
        return img

    def _get_label(self, i):
        label_path = os.path.join(
            self.data_dir, 'SegmentationClass', self.ids[i] + '.png')
        label = read_label(label_path, dtype=np.int32)
        label[label == 255] = -1
        # (1, H, W) -> (H, W)
        return label
    
    def _get_depth(self, i):
        depth_path = os.path.join(
            self.depth_dir, self.ids[i] + '.png')
        
        depth_img = read_image(depth_path, mode = 'L')
        
        return depth_img

    def _get_box_size(self, i):
        cam_path = os.path.join(self.cam_dir, self.ids[i] + ".npy")
        
        cam_dict = np.load(cam_path, allow_pickle=True).item()
        cams = cam_dict['high_res']
        
        box_size = []
        for idx, cam in enumerate(cams):
            conf_cam = find_conf_cam(cam)
            crop_box = find_crop_box(conf_cam)
            size = bbox_size(crop_box)
            
            box_size.append(size)
            
        return box_size
