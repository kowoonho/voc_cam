import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import imageio

import voc12.dataloader
from misc import torchutils, indexing
from tqdm import tqdm

cudnn.enabled = True

def _work(model, dataset, args):

    data_loader = DataLoader(dataset,
                             shuffle=False, num_workers=args.num_workers, pin_memory=False)

    with torch.no_grad():

        model.to(args.device)

        for iter, pack in enumerate(tqdm(data_loader)):
            img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
            orig_img_size = np.asarray(pack['size'])
            edge, dp = model(pack['img'][0].to(args.device))
            
            if args.crop == False:
                cam_dict = np.load(args.cam_out_dir + '/' + img_name + '.npy', allow_pickle=True).item()
            else:
                cam_dict = np.load(args.crop_cam_out_dir + '/' + img_name + '.npy', allow_pickle=True).item()

            cams = cam_dict['cam']
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            cam_downsized_values = cams.to(args.device)

            rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5, device=args.device)
            
            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0], :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up)

            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=args.sem_seg_bg_thres)
            # rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=0)
            rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()

            rw_pred = keys[rw_pred]

            if args.crop==False:
                imageio.imsave(os.path.join(args.sem_seg_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))
            else:
                imageio.imsave(os.path.join(args.crop_sem_seg_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))


def run(args):
    model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')()
    
    if args.crop==False:
        model.load_state_dict(torch.load(args.irn_weights_name+".pth"), strict=False)
    else:
        model.load_state_dict(torch.load(args.crop_irn_weights_name+".pth"), strict=False)
        
    model.eval()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.infer_list,
                                                             voc12_root=args.voc12_root,
                                                             scales=(1.0,))

    _work(model, dataset, args)
