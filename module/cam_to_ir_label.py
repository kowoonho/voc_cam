
import os
import numpy as np
import imageio

from torch import multiprocessing
from torch.utils.data import DataLoader

import voc12.dataloader
from misc import torchutils, imutils
from tqdm import tqdm

def _work(infer_dataset, args):
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=0, pin_memory=False)

    for iter, pack in enumerate(tqdm(infer_data_loader)):
        img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
        img = pack['img'][0].numpy()
        cam_dict = np.load(os.path.join(args.cam_out_dir, img_name + '.npy'), allow_pickle=True).item()
        cams = cam_dict['high_res']
        
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        # 1. find confident fg & bg
        fg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_fg_thres)
        fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
        
        pred = imutils.crf_inference_label(img, fg_conf_cam, n_labels=keys.shape[0])
        fg_conf = keys[pred]

        bg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_bg_thres)
        bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
        pred = imutils.crf_inference_label(img, bg_conf_cam, n_labels=keys.shape[0])
        bg_conf = keys[pred]

        # 2. combine confident fg & bg
        conf = fg_conf.copy()
        conf[fg_conf == 0] = 255
        conf[bg_conf + fg_conf == 0] = 0

        if args.crop==False:
            imageio.imwrite(os.path.join(args.ir_label_out_dir, img_name + '.png'),
                            conf.astype(np.uint8))
        else:
            imageio.imwrite(os.path.join(args.crop_ir_label_out_dir, img_name + '.png'),
                            conf.astype(np.uint8))
        

def run(args):
    dataset = voc12.dataloader.VOC12ImageDataset(args.trainval_list, voc12_root=args.voc12_root, img_normal=None, to_torch=False)
    _work(dataset, args)
    
