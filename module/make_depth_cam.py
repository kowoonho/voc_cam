import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os

import voc12.dataloader
from misc import torchutils, imutils
from tqdm import tqdm
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
            depth_img = pack['depth']
            
            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)
            
            outputs = [model(img[0].to(args.device))
                       for img in pack['img']]
        

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

            valid_cat = torch.nonzero(label)[:, 0]

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5
            
            depth_tensor = (-1 * depth_img + torch.max(depth_img)).to(torch.float32)
            depth_tensor *= args.depth_alpha
            highres_cam = highres_cam.cpu()
            
            highres_depth_cam = depth_tensor + highres_cam
            
            # save cams
            np.save(os.path.join(args.depth_cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "high_res": highres_cam.cpu().numpy()})



def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model.eval()
    
    dataset = voc12.dataloader.VOC12DepthDatasetMSF(args.trainval_list, voc12_root=args.voc12_root,
                                                             depth_root = args.depth_root, scales=args.cam_scales)
    _work(model, dataset, args)