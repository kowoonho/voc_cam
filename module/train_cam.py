
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib

import voc12.dataloader
import voc12.my_dataloader
from misc import pyutils, torchutils
from tqdm import tqdm
import torch.nn as nn
def validate(model, data_loader, args):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()

    with torch.no_grad():
        for pack in tqdm(data_loader):
            img = pack['img'].to(args.device)

            label = pack['label'].to(args.device)

            x = model(img)
            loss1 = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss1': loss1.item()})

    model.train()

    print('loss: %.4f' % (loss1.item()))

    return


def run(args):

    model = getattr(importlib.import_module(args.cam_network), 'Net')(rgbd=args.rgbd)

    if args.crop==False and args.rgbd == False:
        train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                    resize_long=(320, 640), hor_flip=True,
                                                                    crop_size=512, crop_method="random")
        val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    elif args.crop==True and args.rgbd == False:
        train_dataset = voc12.my_dataloader.VOC12_CropImages(args.train_list, voc12_root = args.voc12_root, 
                                                             cam_root = args.cam_root, preprocessing=True)
        val_dataset = voc12.my_dataloader.VOC12_CropImages(args.val_list, voc12_root = args.voc12_root, 
                                                             cam_root = args.cam_root, preprocessing=True)
    else:
        print("RGBD!")
        train_dataset = voc12.my_dataloader.VOC12_DepthClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                             depth_root=args.depth_root,
                                                                             resize_long=(320, 640), hor_flip=True,
                                                                             crop_size=512, crop_method='random')
        val_dataset = voc12.my_dataloader.VOC12_DepthClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                                             depth_root=args.depth_root,
                                                                             resize_long=(320, 640), hor_flip=True,
                                                                             crop_size=512, crop_method='random')
    
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches
    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)
    
    model = model.to(args.device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

        for step, pack in enumerate(tqdm(train_data_loader)):

            img = pack['img'].to(args.device)
            label = pack['label'].to(args.device)
            x = model(img)
            if args.crop:
                label = torch.argmax(label, dim = 1)
                loss1 = criterion(x, label)
            else:
                loss1 = F.multilabel_soft_margin_loss(x, label)
            
            avg_meter.add({'loss1': loss1.item()})

            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()

        print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                'loss:%.4f' % (loss1.item()),
                'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                'lr: %.4f' % (optimizer.param_groups[0]['lr']))

        
        validate(model, val_data_loader, args)
    
    if args.crop == False and args.rgbd == False:
        torch.save(model.state_dict(), args.cam_weights_name + '.pth')
    elif args.crop == True and args.rgbd == False:
        torch.save(model.state_dict(), args.crop_cam_weights_name + '.pth')
    else:
        torch.save(model.state_dict(), args.rgbd_cam_weights_name + '.pth')
        
    
    