import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os

import voc12.dataloader
import voc12.my_dataloader
from misc import torchutils, imutils
from tqdm import tqdm
from utility import image_util
cudnn.enabled = True

def _work(model, dataset, args):
    data_loader = DataLoader(dataset, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    with torch.no_grad():
        model = model.to(args.device)

        for iter, pack in enumerate(tqdm(data_loader)):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']
            image = pack['img'] # 4 MSF images [2, 3, w, h]
            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)
            
            outputs = [model(img[0].to(args.device))
                       for img in pack['img']]
            
            
            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs]), 0)
        

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5
            
            # save cams
            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})
            

def crop_work(model, dataset, args):
    data_loader = DataLoader(dataset, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    with torch.no_grad():
        model = model.to(args.device)

        for iter, pack in enumerate(tqdm(data_loader)):

            img_name = pack['name'][0]
            msf = pack['msf_img_list']
            org_size = pack['img'].shape[1:]

            crop_labels = pack['crop_labels']
            crop_boxes = pack['crop_boxes']

            cam_list = []

            key = torch.sum(torch.cat(crop_labels, dim=0), dim=0)
            key = torch.nonzero(key)[:, 0]
            for idx, msf_img in enumerate(msf):
                
                msf_img = msf[idx]
                label = crop_labels[idx][0]
                
                size = (crop_boxes[idx][3] - crop_boxes[idx][1], crop_boxes[idx][2] - crop_boxes[idx][0])
                
                strided_up_size = imutils.get_strided_up_size(size, 16)
                
                valid_cat = torch.nonzero(label)[:, 0]
                
                outputs = [model(img[0].to(args.device)) for img in msf_img]
                
                highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                            mode='bilinear', align_corners=False) for o in outputs]

                
                highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
                
                highres_cam = highres_cam[valid_cat]
                highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5
                highres_cam = highres_cam[0].cpu().numpy()
                highres_cam = image_util.crop_cam_to_org_cam(highres_cam, crop_boxes[idx], org_size)
                
                cam_list.append(highres_cam)
                
            cam_stack = np.stack(cam_list)

            np.save(os.path.join(args.crop_cam_out_dir, img_name + '.npy'),
                    {"keys": key, "cams": cam_stack})



def run(args, crop=False):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    if crop == False:
        model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
        
    else:
        model.load_state_dict(torch.load(args.crop_cam_weights_name + '.pth'), strict=True)
        
    model.eval()
    
    if crop == False:
        dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.trainval_list,
                                                                 voc12_root=args.voc12_root, scales=args.cam_scales)
        _work(model, dataset, args)
        
    else:
        dataset = voc12.my_dataloader.VOC12_CropClassificationDatasetMSF(args.trainval_list, voc12_root = args.voc12_root,
                                                          cam_root = args.cam_root, scales=args.cam_scales)
        crop_work(model, dataset, args)