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
            if args.rgbd:
                np.save(os.path.join(args.rgbd_cam_out_dir, img_name + '.npy'),
                        {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})
            else:
                np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                        {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})
            

def crop_work(model, dataset, args):
    data_loader = DataLoader(dataset, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    with torch.no_grad():
        model = model.to(args.device)

        for iter, pack in enumerate(tqdm(data_loader)):

            img_name = pack['name'][0]
            msf = pack['msf_img_list']
            org_size = pack['size']
            strided_org_size = imutils.get_strided_size(org_size, 4)

            org_cam = pack['cam'][0].cpu().numpy()
            org_high_res = pack['high_res'][0].cpu().numpy()
            crop_labels = pack['crop_labels']
            crop_boxes = pack['crop_boxes']
            crop_boxes = [tuple(int(t.item()) for t in crop_box) for crop_box in crop_boxes]
            
            strided_crop_boxes = image_util.resize_bbox_list(crop_boxes, org_size, strided_org_size)
            
            cam_list = []
            highres_cam_list = []

            key = torch.sum(torch.cat(crop_labels, dim=0), dim=0)
            key = torch.nonzero(key)[:, 0]
            for idx, msf_img in enumerate(msf):
                
                msf_img = msf[idx]
                label = crop_labels[idx][0]
                
                size = (crop_boxes[idx][3] - crop_boxes[idx][1], crop_boxes[idx][2] - crop_boxes[idx][0])
                
                strided_size = imutils.get_strided_size(size, 4)
                strided_up_size = imutils.get_strided_up_size(size, 16)
                
                valid_cat = torch.nonzero(label)[:, 0]
                
                outputs = [model(img[0].to(args.device)) for img in msf_img]
                
                strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                    in outputs]), 0)
                
                highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                            mode='bilinear', align_corners=False) for o in outputs]

                
                highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
                
                highres_cam = highres_cam[valid_cat]
                highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5
                
                strided_cam = strided_cam[valid_cat]
                strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5
                
                strided_cam = strided_cam[0].cpu().numpy()
                highres_cam = highres_cam[0].cpu().numpy()

                strided_cam = image_util.crop_cam_to_org_cam(strided_cam, strided_crop_boxes[idx], strided_org_size)
                highres_cam = image_util.crop_cam_to_org_cam(highres_cam, crop_boxes[idx], org_size)
                
                cam_list.append(strided_cam)
                highres_cam_list.append(highres_cam)
            
            cam_stack = np.stack(cam_list)
            cam_stack = (cam_stack + org_cam) / 2

            highres_cam_stack = np.stack(highres_cam_list)
            highres_cam_stack = (highres_cam_stack + org_high_res) / 2
            
            cam_stack = torch.from_numpy(cam_stack)

            np.save(os.path.join(args.crop_cam_out_dir, img_name + '.npy'),
                    {"keys": key, "cam" : cam_stack, "high_res": highres_cam_stack})
            
def depth_crop_work(model, dataset, args):
    data_loader = DataLoader(dataset, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    with torch.no_grad():
        model = model.to(args.device)

        for iter, pack in enumerate(tqdm(data_loader)):

            img_name = pack['name'][0]
            msf = pack['msf_img_list']
            org_size = pack['size']
            strided_org_size = imutils.get_strided_size(org_size, 4)

            org_cam = pack['cam'][0].cpu().numpy()
            org_high_res = pack['high_res'][0].cpu().numpy()
            crop_labels = pack['crop_labels']
            crop_boxes = pack['crop_boxes']
            crop_sizes = pack['crop_sizes']
            scale_crop_sizes = pack['scale_crop_sizes']
            crop_boxes = [tuple(int(t.item()) for t in crop_box) for crop_box in crop_boxes]
            
            strided_crop_boxes = image_util.resize_bbox_list(crop_boxes, org_size, strided_org_size)
            
            cam_list = []
            highres_cam_list = []

            key = torch.sum(torch.cat(crop_labels, dim=0), dim=0)
            key = torch.nonzero(key)[:, 0]
            for idx, msf_img in enumerate(msf):
                
                msf_img = msf[idx]
                label = crop_labels[idx][0]
                
                scale_crop_size = scale_crop_sizes[idx]
                org_crop_size = crop_sizes[idx]
                
                scale_strided_size = imutils.get_strided_size(scale_crop_size, 4)
                scale_strided_up_size = imutils.get_strided_up_size(scale_crop_size, 16)
                
                org_strided_size = imutils.get_strided_size(org_crop_size, 4)
                
                valid_cat = torch.nonzero(label)[:, 0]
                
                outputs = [model(img[0].to(args.device)) for img in msf_img]
                
                scale_strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), scale_strided_size, mode='bilinear', align_corners=False)[0] for o
                    in outputs]), 0)
                
                scale_highres_cam = [F.interpolate(torch.unsqueeze(o, 1), scale_strided_up_size,
                                            mode='bilinear', align_corners=False) for o in outputs]

                
                scale_highres_cam = torch.sum(torch.stack(scale_highres_cam, 0), 0)[:, 0, :scale_crop_size[0], :scale_crop_size[1]]
                
                scale_highres_cam = scale_highres_cam[valid_cat]
                scale_highres_cam /= F.adaptive_max_pool2d(scale_highres_cam, (1, 1)) + 1e-5
                
                scale_strided_cam = scale_strided_cam[valid_cat]
                scale_strided_cam /= F.adaptive_max_pool2d(scale_strided_cam, (1, 1)) + 1e-5
                
                scale_strided_cam = scale_strided_cam[0].cpu().numpy()
                scale_highres_cam = scale_highres_cam[0].cpu().numpy()
                
                print(org_strided_size)
                print(scale_strided_cam.shape)
                print(scale_crop_size[idx])
                print(org_crop_size[idx])
                
                
                

                # strided_cam = image_util.crop_cam_to_org_cam(strided_cam, strided_crop_boxes[idx], strided_org_size)
            #     highres_cam = image_util.crop_cam_to_org_cam(highres_cam, crop_boxes[idx], org_size)
                
            #     cam_list.append(strided_cam)
            #     highres_cam_list.append(highres_cam)
            
            # cam_stack = np.stack(cam_list)
            # cam_stack = (cam_stack + org_cam) / 2

            # highres_cam_stack = np.stack(highres_cam_list)
            # highres_cam_stack = (highres_cam_stack + org_high_res) / 2
            
            # cam_stack = torch.from_numpy(cam_stack)

            # np.save(os.path.join(args.depth_crop_cam_out_dir, img_name + '.npy'),
            #         {"keys": key, "cam" : cam_stack, "high_res": highres_cam_stack})

def grid_crop_work(model, dataset, args):
    data_loader = DataLoader(dataset, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    with torch.no_grad():
        model = model.to(args.device)

        for iter, pack in enumerate(tqdm(data_loader)):

            img_name = pack['name'][0]
            label = pack['label'][0]
            org_size = pack['size']
            size = pack['crop_size']
            msf = pack['msf']
            
            org_strided_size = imutils.get_strided_size(org_size, 4)
            
            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)
            valid_cat = torch.nonzero(label)[:, 0]
            
            strided_cam_list = []
            highres_cam_list = []
            
            for i, msf_img in enumerate(msf):
                outputs = [model(img[0].to(args.device)) for img in msf_img]
                
                strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs]), 0)

                highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                            mode='bilinear', align_corners=False) for o in outputs]
                highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

                strided_cam = strided_cam[valid_cat]
                strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

                highres_cam = highres_cam[valid_cat]
                highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5
                
                strided_cam = strided_cam.cpu().numpy()
                highres_cam = highres_cam.cpu().numpy()
                
                strided_cam_list.append(strided_cam)
                highres_cam_list.append(highres_cam)
                
                
            strided_cam = image_util.merge_images(strided_cam_list, org_strided_size)
            highres_cam = image_util.merge_images(highres_cam_list, org_size)
            
            strided_cam = torch.from_numpy(strided_cam)
            

            # save cams
            np.save(os.path.join(args.grid_cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam, "high_res": highres_cam})
            

def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')(args.rgbd)
    if args.rgbd == False:
        if args.crop == False or args.grid == True:
            model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
        
        else:
            model.load_state_dict(torch.load(args.crop_cam_weights_name + '.pth'), strict=True)
    else:
        model.load_state_dict(torch.load(args.rgbd_cam_weights_name + '.pth'), strict=True)
    model.eval()
    
    
    if args.rgbd == False:
        if args.crop == False and args.grid == False:
            dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.trainval_list,
                                                                    voc12_root=args.voc12_root, scales=args.cam_scales)
            _work(model, dataset, args)
            
        elif args.grid==True:
            print("Make Grid Crop Cam!")
            dataset = voc12.my_dataloader.VOC12_GridCropImageDatasetMSF(args.trainval_list, voc12_root = args.voc12_root,
                                                                        scales = args.cam_scales)
            grid_crop_work(model, dataset, args)
            
        else:
            dataset = voc12.my_dataloader.VOC12_Depth_CropClassificationDatasetMSF(args.trainval_list, voc12_root = args.voc12_root,
                                                            cam_root = args.cam_root, depth_root = args.depth_root,
                                                            scales=args.cam_scales)
            depth_crop_work(model, dataset, args)
    
    else:
        print("RGBD!")
        dataset = voc12.my_dataloader.VOC12_DepthClassificationDatasetMSF(args.trainval_list, voc12_root= args.voc12_root,
                                                                          depth_root=args.depth_root, scales = args.cam_scales)
        _work(model, dataset, args)
        
    