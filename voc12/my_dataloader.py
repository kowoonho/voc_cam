import numpy as np
import torch
from torch.utils.data import Dataset
import os.path
import imageio
from misc import imutils
from torchvision import transforms
from PIL import Image
import cv2
from utility import image_util, util
from torchvision.transforms import RandomCrop
from tqdm import tqdm

IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"
IGNORE = 255

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

cls_labels_dict = np.load('voc12/cls_labels.npy', allow_pickle=True).item()

def decode_int_filename(int_filename):
    s = str(int(int_filename))
    return s[:4] + '_' + s[4:]

def decode_str_filename(str_filename):
    int_filename = int(str_filename[:4]+str_filename[5:])
    return int_filename


def load_img_id_list(img_id_file):
    return open(img_id_file).read().splitlines()

def load_image_label_list_from_npy(img_name_list):
    if isinstance(img_name_list[0], str):
        return np.array([cls_labels_dict[decode_str_filename(img_name)] for img_name in img_name_list])
    else:
        return np.array([cls_labels_dict[img_name] for img_name in img_name_list])



class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img
    
class TorchvisionNormalizeRGBD():
    def __init__(self, mean=(0.485, 0.456, 0.406, 0.5), std=(0.229, 0.224, 0.225, 0.5)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img, dtype=np.float32)
        proc_img = np.empty_like(imgarr)

        for i in range(3):
            proc_img[..., i] = (imgarr[..., i] / 255. - self.mean[i]) / self.std[i]

        proc_img[..., 3] = (imgarr[..., 3] / 255. - self.mean[3]) / self.std[3]

        return proc_img
    
def preprocessing(img, resize_long=(320, 640), rescale = None, img_normal = TorchvisionNormalize(),
                  hor_flip = True, crop_size = 512, to_torch=True):
    if resize_long:
            img = imutils.random_resize_long(img, resize_long[0], resize_long[1])

    if rescale:
        img = imutils.random_scale(img, scale_range=rescale, order=3)

    if img_normal:
        img = img_normal(img)

    if hor_flip:
        img = imutils.random_lr_flip(img)

    if crop_size:
        img = imutils.random_crop(img, crop_size, 0)

    if to_torch:
        img = imutils.HWC_to_CHW(img)
        
    return img
    
class VOC12_Dataset(Dataset):
    def __init__(self, img_name_list_path, voc12_root):
        self.img_name_list = load_img_id_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
    
    def __len__(self):
        return len(self.img_name_list)
    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = image_util.read_image(os.path.join(self.voc12_root, "JPEGImages", name+".jpg"))
        label = self.label_list[idx]

        return {'name' : name, 'img' : img, 'label' : label}
            
class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True):

        self.img_name_list = load_img_id_list(img_name_list_path)
        self.voc12_root = voc12_root

        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = np.asarray(imageio.imread(os.path.join(self.voc12_root, "JPEGImages", name+".jpg")))

        if self.resize_long:
            img = imutils.random_resize_long(img, self.resize_long[0], self.resize_long[1])

        if self.rescale:
            img = imutils.random_scale(img, scale_range=self.rescale, order=3)

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img = imutils.random_lr_flip(img)

        if self.crop_size:
            if self.crop_method == "random":
                img = imutils.random_crop(img, self.crop_size, 0)
            else:
                img = imutils.top_left_crop(img, self.crop_size, 0)

        if self.to_torch:
            img = imutils.HWC_to_CHW(img)

        return {'name': name, 'img': img}

class VOC12ClassificationDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None):
        super().__init__(img_name_list_path, voc12_root,
                 resize_long, rescale, img_normal, hor_flip,
                 crop_size, crop_method)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        out['label'] = torch.from_numpy(self.label_list[idx])

        return out

class VOC12ClassificationDatasetMSF(VOC12ClassificationDataset):

    def __init__(self, img_name_list_path, voc12_root,
                 img_normal=TorchvisionNormalize(),
                 scales=(1.0,)):
        self.scales = scales

        super().__init__(img_name_list_path, voc12_root, img_normal=img_normal)
        self.scales = scales
    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = imageio.imread(os.path.join(self.voc12_root, 'JPEGImages', name+".jpg"))

        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
            s_img = self.img_normal(s_img)
            s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]

        out = {"name": name, "img": ms_img_list, "size": (img.shape[0], img.shape[1]),
               "label": torch.from_numpy(self.label_list[idx])}
        return out
    
class VOC12CamDataset(VOC12_Dataset):
    def __init__(self, img_name_list_path, voc12_root, cam_root):
        super().__init__(img_name_list_path, voc12_root)
        self.cam_root = cam_root
    
    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        name = out['name']
        cam_dict= np.load(os.path.join(self.cam_root, name + ".npy"), allow_pickle=True).item()

        out['cam'] = cam_dict['cam']
        out['high_res'] = cam_dict['high_res']
        
        return out
    
class VOC12CropImageDataset(VOC12CamDataset):
    def __init__(self, img_name_list_path, voc12_root, cam_root, crop_resize=None):
        super().__init__(img_name_list_path, voc12_root, cam_root)
    
        self.crop_resize = crop_resize
        self.calculate_len()
        
    def calculate_len(self):
        self.cam_len_list = []
        len_val = 0
        self.cam_len_list.append(len_val)
        for idx in tqdm(range(len(self.img_name_list))):
            
            out = super().__getitem__(idx)
            len_val += len(out['cam'])
            self.cam_len_list.append(len_val)
            
        self.cam_len_list = np.asarray(self.cam_len_list)
    
    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        
        label = out['label']
        cam = out['cam']
        high_res = out['high_res']
        img = out['img']
        name = out['name']
        size = img.shape[:2]
        
        crop_images = []
        crop_labels = []
        crop_boxes = []
        
        label_cat = np.where(label == 1)[0]
        for i in range(len(high_res)):
            
            conf_cam = util.find_conf_cam(high_res[i])
            crop_box = util.find_crop_box(conf_cam)
            cropped_img = util.crop_image_with_bounding_box(img, crop_box)
            if self.crop_resize:
                cropped_img = cv2.resize(cropped_img, self.crop_resize)
    
            cropped_label = np.zeros(20)
            cropped_label[label_cat[i]] = 1
            crop_boxes.append(crop_box)
            crop_images.append(cropped_img)
            crop_labels.append(cropped_label)
        
        return {'name' : name, 'img' : img, 'label' : label, 'crop_images' : crop_images,
                'crop_labels' : crop_labels, 'crop_boxes' : crop_boxes, 'cam' : cam, 'high_res' : high_res,
                'size' : size}
        
class VOC12_CropImages(VOC12CropImageDataset):
    def __init__(self, img_name_list_path, voc12_root, cam_root, crop_resize = None, preprocessing = False):
        super().__init__(img_name_list_path, voc12_root, cam_root, crop_resize)
        self.preprocessing = preprocessing
        print("Total Image : {}".format(len(self)))
    def __len__(self):
        return self.cam_len_list[len(self.cam_len_list) - 1]

    def find_index(self, idx):
        prev_value_index = np.argmax(self.cam_len_list > idx) - 1
        return prev_value_index
    
    def __getitem__(self, idx):
        real_idx = self.find_index(idx)
        out = super().__getitem__(real_idx)

        crop_idx = self.cam_len_list[real_idx] - idx

        
        name = "{}_{}".format(out['name'], crop_idx)
        img = out['crop_images'][crop_idx]
        label = out['crop_labels'][crop_idx]
        label = torch.from_numpy(label)
        
        if self.preprocessing:
            img = preprocessing(img)
        
        return {'name' : name, 'img' : img, 'label' : label}

class VOC12_CropClassificationDatasetMSF(VOC12CropImageDataset):
    def __init__(self, img_name_list_path, voc12_root,
                 cam_root, img_normal = TorchvisionNormalize(), scales=(1.0,)):
        self.scales = scales
        self.img_normal = img_normal
        super().__init__(img_name_list_path, voc12_root, cam_root)
    
    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        
        crop_images = out['crop_images']
        
        msf_img_list = []
        
        for crop_img in crop_images:
            ms_img_list = []
            
            for s in self.scales:
                if s == 1:
                    s_img = crop_img
                    
                else:
                    s_img = imutils.pil_rescale(crop_img, s, order=3)

                if self.img_normal:
                    s_img = self.img_normal(s_img)
                s_img = imutils.HWC_to_CHW(s_img)
                ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis = 0))
            if len(self.scales) == 1:
                ms_img_list = ms_img_list[0]
                
            msf_img_list.append(ms_img_list)
            
        out['msf_img_list'] = msf_img_list

        return out
        
class VOC12_GridCropImageDataset(VOC12_Dataset):
    def __init__(self, img_name_list_path, voc12_root, crop_resize=None, 
                 img_normal=TorchvisionNormalize(), to_torch=True):
        super().__init__(img_name_list_path, voc12_root)
        self.crop_resize = crop_resize
        self.img_normal = img_normal
        self.to_torch = to_torch
        
    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        
        name = out['name']
        img = out['img']
        
        images = image_util.crop_image(img)
        
        out['crop_images'] = images
        
        return out
    
class VOC12_GridCropImageDatasetMSF(VOC12_GridCropImageDataset):
    def __init__(self, img_name_list_path, voc12_root, crop_resize=None, scales=(1.0,)):
        self.scales = scales
        
        super().__init__(img_name_list_path, voc12_root, crop_resize)
        
    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        
        img = out['img']
        size = img.shape[:2]
        crop_images = out['crop_images']
        crop_size = crop_images[0].shape[:2]
        msf_img_list = []
        
        for crop_img in crop_images:
            ms_img_list = []
            for s in self.scales:
                if s == 1:
                    s_img = crop_img
                    
                else:
                    s_img = imutils.pil_rescale(crop_img, s, order=3)

                if self.img_normal:
                    s_img = self.img_normal(s_img)
                s_img = imutils.HWC_to_CHW(s_img)
                ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis = 0))
            if len(self.scales) == 1:
                ms_img_list = ms_img_list[0]
                
            msf_img_list.append(ms_img_list)
            
        out['msf'] = msf_img_list
        out['size'] = size
        out['crop_size'] = crop_size
        
        return out
        
        
class VOC12_Depth_Crop(VOC12CamDataset):
    def __init__(self, img_name_list_path, voc12_root, cam_root, depth_root):
        super().__init__(img_name_list_path, voc12_root, cam_root)
        
        self.depth_root = depth_root
        
    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        label = out['label']
        high_res = out['high_res']
        img = out['img']
        name = out['name']
        out['size'] = img.shape[:2]
        
        depth_img = image_util.read_image(os.path.join(self.depth_root, name + ".png"))
        
        crop_images = []
        crop_labels = []
        crop_boxes = []
        mean_depths = []
        crop_sizes = []
        
        label_cat = np.where(label == 1)[0]
        for i in range(len(high_res)):
            
            conf_cam = util.find_conf_cam(high_res[i])
            crop_box = util.find_crop_box(conf_cam)
            cropped_img = util.crop_image_with_bounding_box(img, crop_box)
            mean_depth = util.cam_depth(conf_cam, depth_img)
            
            cropped_label = np.zeros(20)
            cropped_label[label_cat[i]] = 1
            
            crop_sizes.append(cropped_img.shape[:2])
            crop_boxes.append(crop_box)
            crop_images.append(cropped_img)
            crop_labels.append(cropped_label)
            mean_depths.append(mean_depth)
        
        out['mean_depths'] = mean_depths
        out['crop_images'] = crop_images
        out['crop_labels'] = crop_labels
        out['crop_boxes'] = crop_boxes
        out['crop_sizes'] = crop_sizes
        return out
    
class VOC12_Depth_CropClassificationDatasetMSF(VOC12_Depth_Crop):
    def __init__(self, img_name_list_path, voc12_root, cam_root, depth_root, scales = (1.0,)):
        super().__init__(img_name_list_path, voc12_root, cam_root, depth_root)
        self.scales = scales
        
        self.img_normal = TorchvisionNormalize()

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        
        msf_img_list = []

        scale_crop_sizes = []
        
        for i, crop_img in enumerate(out['crop_images']):
            ms_img_list = []
            depth_scale_factor = util.depth_scaling(out['mean_depths'][i])
            
            depth_scale_img = imutils.pil_rescale(crop_img, depth_scale_factor, order=3)
            
            scale_crop_sizes.append(depth_scale_img.shape[:2])
            for s in self.scales:
                if s == 1:
                    s_img = depth_scale_img
                else:
                    s_img = imutils.pil_rescale(depth_scale_img, s, order=3)

                if self.img_normal:
                    s_img = self.img_normal(s_img)
                s_img = imutils.HWC_to_CHW(s_img)
                ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis = 0))
            if len(self.scales) == 1:
                ms_img_list = ms_img_list[0]
                
            msf_img_list.append(ms_img_list)
        
        out['scale_crop_sizes'] = scale_crop_sizes
        out['msf_img_list'] = msf_img_list
        
        return out
        
        
    
class VOC12_DepthDataset(VOC12_Dataset):
    def __init__(self, img_name_list_path, voc12_root, depth_root):
        super().__init__(img_name_list_path, voc12_root)
        self.depth_root = depth_root
        
    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        depth_img = image_util.read_image(os.path.join(self.depth_root, out['name'] + ".png"))
        
        out['depth'] = depth_img
        
        return out

class VOC12_DepthClassificationDataset(VOC12_DepthDataset):
    def __init__(self, img_name_list_path, voc12_root, depth_root, resize_long=None, rescale=None,
                 img_normal=TorchvisionNormalizeRGBD(), hor_flip = False, crop_size=None, crop_method=None, to_torch=True):
        super().__init__(img_name_list_path, voc12_root, depth_root)
        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch
        
    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        depth_expanded = np.expand_dims(out['depth'], axis=-1)
        rgbd_img = np.concatenate((out['img'], depth_expanded), axis = -1)
        
        
        if self.resize_long:
            rgbd_img = imutils.random_resize_long(rgbd_img, self.resize_long[0], self.resize_long[1])
        
        if self.rescale:
            rgbd_img = imutils.random_scale(rgbd_img, scale_range=self.rescale, order=3)
        
        if self.img_normal:
            rgbd_img = self.img_normal(rgbd_img)
        
        if self.hor_flip:
            rgbd_img = imutils.random_lr_flip(rgbd_img)
            
        if self.crop_size:
            if self.crop_method == 'random':
                rgbd_img = imutils.random_crop(rgbd_img, self.crop_size, 0)
            else:
                rgbd_img = imutils.top_left_crop(rgbd_img, self.crop_size, 0)
        
        if self.to_torch:
            rgbd_img = imutils.HWC_to_CHW(rgbd_img)
        
        label = torch.from_numpy(out['label'])
        
        return {'name' : out['name'], 'img' : rgbd_img, 'label' : label}
        

class VOC12_DepthClassificationDatasetMSF(VOC12_DepthClassificationDataset):
    def __init__(self, img_name_list_path, voc12_root, depth_root, img_normal=TorchvisionNormalizeRGBD(),
                 scales=(1.0,)):
        self.scales = scales
        super().__init__(img_name_list_path, voc12_root, depth_root, img_normal=img_normal)
        
    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = image_util.read_image(os.path.join(self.voc12_root, "JPEGImages", name+".jpg"))
        depth = image_util.read_image(os.path.join(self.depth_root, name+".png"))
        rgbd = np.concatenate((img, np.expand_dims(depth, axis = -1)), axis = -1)
        

        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = rgbd
            else:
                s_img = imutils.pil_rescale(rgbd, s, order=3)
            s_img = self.img_normal(s_img)
            s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]

        out = {"name": name, "img": ms_img_list, "size": (img.shape[0], img.shape[1]),
               "label": torch.from_numpy(self.label_list[idx])}
        return out