import argparse
import os
import psutil


from misc import pyutils
def main_process():
    try:
        parser = argparse.ArgumentParser()

        # Environment
        parser.add_argument("--num_workers", default=os.cpu_count(), type=int)
        parser.add_argument("--dataset", default="voc2012", type=str)

        # Dataset
        parser.add_argument("--train_list", default="voc12/train.txt", type=str)
        parser.add_argument("--val_list", default="voc12/val.txt", type=str)
        parser.add_argument("--trainval_list", default="voc12/trainval.txt", type=str)
        parser.add_argument("--voc12_root", default="../Dataset/VOC2012", type=str)
        parser.add_argument("--depth_root", default="../result/depth_img")
        parser.add_argument("--cam_root", default="../irn_result/cam")
        parser.add_argument("--infer_list", default="voc12/val.txt", type=str,
                            help="voc12/train_aug.txt to train a fully supervised model, "
                                "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
        parser.add_argument("--chainer_eval_set", default="train", type=str)
        parser.add_argument("--rgbd", default=False, type=str)
        

        # Class Activation Map
        parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
        parser.add_argument("--cam_batch_size", default=32, type=int)
        parser.add_argument("--cam_num_epoches", default=5, type=int)
        parser.add_argument("--cam_learning_rate", default=0.05, type=float)
        parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
        parser.add_argument("--cam_eval_thres", default=0.15, type=float)
        parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                            help="Multi-scale inferences")

        # Mining Inter-pixel Relations
        parser.add_argument("--conf_fg_thres", default=0.30, type=float)
        parser.add_argument("--conf_bg_thres", default=0.05, type=float)

        # Inter-pixel Relation Network (IRNet)
        parser.add_argument("--irn_network", default="net.resnet50_irn", type=str)
        parser.add_argument("--irn_crop_size", default=512, type=int)
        parser.add_argument("--irn_batch_size", default=16, type=int)
        parser.add_argument("--irn_num_epoches", default=5, type=int)
        parser.add_argument("--irn_learning_rate", default=0.05, type=float)
        parser.add_argument("--irn_weight_decay", default=1e-4, type=float)
        
        parser.add_argument("--crop_resize", default=(512, 512))
        

        # Random Walk Params
        parser.add_argument("--beta", default=10)
        parser.add_argument("--exp_times", default=8,
                            help="Hyper-parameter that controls the number of random walk iterations,"
                                "The random walk is performed 2^{exp_times}.")
        parser.add_argument("--ins_seg_bg_thres", default=0.25)
        parser.add_argument("--sem_seg_bg_thres", default=0.25)

        # Output Path
        parser.add_argument("--log_name", default="sample_train_eval", type=str)
        parser.add_argument("--cam_weights_name", default="../sess/voc_sess/resnet50_cam", type=str)
        parser.add_argument("--crop_cam_weights_name", default="../sess/voc_sess/resnet50_crop_cam", type=str)
        parser.add_argument("--depth_crop_cam_weights_name", default="../sess/voc_sess/resnet50_depth_crop_cam", type=str)
        parser.add_argument("--rgbd_cam_weights_name", default="../sess/voc_sess/resnet50_rgbd_cam", type=str)
        
        parser.add_argument("--irn_weights_name", default="../sess/voc_sess/resnet50_irn", type=str)
        parser.add_argument("--crop_irn_weights_name", default="../sess/voc_sess/resnet50_crop_irn", type=str)
        parser.add_argument("--depth_crop_irn_weights_name", default="../sess/voc_sess/resnet50_depth_crop_irn", type=str)
        
        
        parser.add_argument("--depth_crop_cam_out_dir", default="../irn_result/depth_crop_cam", type=str)
        parser.add_argument("--rgbd_cam_out_dir", default="../irn_result/rgbd_cam", type=str)
        parser.add_argument("--grid_cam_out_dir", default="../irn_result/grid_cam", type=str)
        parser.add_argument("--crop_cam_out_dir", default="../irn_result/crop_cam", type=str)
        parser.add_argument("--cam_out_dir", default="../irn_result/cam", type=str)
        
        parser.add_argument("--ir_label_out_dir", default="../irn_result/ir_label", type=str)
        parser.add_argument("--crop_ir_label_out_dir", default="../irn_result/crop_ir_label", type=str)
        parser.add_argument("--depth_crop_ir_label_out_dir", default="../irn_result/depth_crop_ir_label", type=str)
        
        parser.add_argument("--sem_seg_out_dir", default="../irn_result/sem_seg", type=str)
        parser.add_argument("--crop_sem_seg_out_dir", default="../irn_result/crop_sem_seg", type=str)
        parser.add_argument("--depth_crop_sem_seg_out_dir", default="../irn_result/depth_crop_sem_seg", type=str)
        parser.add_argument("--edge_sem_seg_out_dir", default="../irn_result/edge_sem_seg", type=str)
        parser.add_argument("--edge_out_dir", default="../result/edge_map", type=str)
        

        

        # step
        parser.add_argument("--train_cam_pass", default=False)
        parser.add_argument("--make_cam_pass", default=True)
        parser.add_argument("--eval_cam_pass", default=False)
        parser.add_argument("--cam_to_ir_label_pass", default=True)
        parser.add_argument("--train_irn_pass", default=True)
        parser.add_argument("--make_sem_seg_pass", default=True)
        parser.add_argument("--eval_sem_seg_pass", default=True)
        
        
        parser.add_argument("--crop", default=False)
        parser.add_argument("--grid", default=False)
        parser.add_argument("--edge", default=False)
        parser.add_argument("--depth", default=True)
        

        # device
        parser.add_argument("--device", default="cuda:0", type=str)
        
        args = parser.parse_args()

        os.makedirs("../sess", exist_ok=True)
        os.makedirs(args.rgbd_cam_out_dir, exist_ok=True)
        os.makedirs(args.grid_cam_out_dir, exist_ok=True)
        os.makedirs(args.crop_cam_out_dir, exist_ok=True)
        os.makedirs(args.cam_out_dir, exist_ok=True)
        os.makedirs(args.depth_crop_cam_out_dir, exist_ok=True)
        
        
        os.makedirs(args.ir_label_out_dir, exist_ok=True)
        os.makedirs(args.crop_ir_label_out_dir, exist_ok=True)
        os.makedirs(args.depth_crop_ir_label_out_dir, exist_ok=True)
        
        os.makedirs(args.sem_seg_out_dir, exist_ok=True)
        os.makedirs(args.crop_sem_seg_out_dir, exist_ok=True)
        os.makedirs(args.depth_crop_sem_seg_out_dir, exist_ok=True)
        os.makedirs(args.edge_sem_seg_out_dir, exist_ok=True)
        os.makedirs(args.edge_out_dir, exist_ok=True)
        
    
        

        pyutils.Logger(args.log_name + '.log')
        print(vars(args))

        if args.train_cam_pass:
            import module.train_cam
            
            module.train_cam.run(args)
            
        if args.make_cam_pass:
            import module.make_cam
            
            module.make_cam.run(args)
            
        if args.eval_cam_pass:
            import module.eval_cam
            
            module.eval_cam.run(args)
            
        if args.cam_to_ir_label_pass:
            import module.cam_to_ir_label
            
            module.cam_to_ir_label.run(args)
            
        if args.train_irn_pass:
            import module.train_irn
            
            module.train_irn.run(args)
            
        if args.make_sem_seg_pass:
            import module.make_sem_seg_labels
            
            module.make_sem_seg_labels.run(args)
            
        if args.eval_sem_seg_pass:
            import module.eval_sem_seg
            
            module.eval_sem_seg.run(args)
            
        
    except KeyboardInterrupt:
        print("Interrupted by keyboard. Cleaning up...")
        for proc in psutil.process_iter():
            print(proc.name())
            
            
            


if __name__ == '__main__':
    main_process()

        
