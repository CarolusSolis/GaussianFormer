try:
    from vis import save_occ, save_gaussian, save_gaussian_topdown
except:
    print('Load Occupancy Visualization Tools Failed.')
import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist

from PIL import Image
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load configs for both models
    cfg1 = Config.fromfile(args.py_config1)
    cfg1.work_dir = args.work_dir1
    
    cfg2 = Config.fromfile(args.py_config2)
    cfg2.work_dir = args.work_dir2

    # Create a common output directory
    common_save_dir = args.common_save_dir
    os.makedirs(common_save_dir, exist_ok=True)
    
    # init DDP
    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20507")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg1.gpu_ids = range(world_size)
        cfg2.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1
    
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(common_save_dir, f'{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    logger.info(f'Config 1:\n{cfg1.pretty_text}')
    logger.info(f'Config 2:\n{cfg2.pretty_text}')

    # build models
    import model
    from dataset import get_dataloader

    # Build and initialize model 1
    model1 = build_segmentor(cfg1.model)
    model1.init_weights()
    n_parameters1 = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    logger.info(f'Model 1 - Number of params: {n_parameters1}')
    
    # Build and initialize model 2
    model2 = build_segmentor(cfg2.model)
    model2.init_weights()
    n_parameters2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    logger.info(f'Model 2 - Number of params: {n_parameters2}')

    if distributed:
        if cfg1.get('syncBN', True):
            model1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model1)
            model2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model2)
            logger.info('converted sync bn.')

        find_unused_parameters1 = cfg1.get('find_unused_parameters', False)
        find_unused_parameters2 = cfg2.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        
        model1 = ddp_model_module(
            model1.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters1)
        raw_model1 = model1.module
        
        model2 = ddp_model_module(
            model2.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters2)
        raw_model2 = model2.module
    else:
        model1 = model1.cuda()
        raw_model1 = model1
        
        model2 = model2.cuda()
        raw_model2 = model2
    
    logger.info('done ddp model')

    # We'll use the same validation config for both models to ensure they visualize the same samples
    val_dataset_config = cfg1.val_dataset_config.copy()
    val_dataset_config.update({
        "vis_indices": args.vis_index,
        "num_samples": args.num_samples,
        "vis_scene_index": args.vis_scene_index})
    
    # Use the same dataloader for both models to ensure they visualize the same samples
    _, val_dataset_loader = get_dataloader(
        cfg1.train_dataset_config,
        val_dataset_config,
        cfg1.train_loader,
        cfg1.val_loader,
        dist=distributed,
        val_only=True)
    
    # Resume and load model 1
    model1_path = args.resume_from1
    logger.info(f'Model 1 resume from: {model1_path}')
    logger.info(f'Model 1 work dir: {args.work_dir1}')

    if model1_path and osp.exists(model1_path):
        map_location = 'cpu'
        ckpt = torch.load(model1_path, map_location=map_location)
        try:
            raw_model1.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)
            logger.info("Model 1 loaded successfully")
        except Exception as e:
            logger.info(f"Error loading model 1: {e}")
            if args.model1_type == "base":
                os.system(f"python modify_weight.py --work-dir {args.work_dir1} --epoch {args.epoch}")
                ckpt = torch.load(model1_path, map_location=map_location)
                raw_model1.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)
                logger.info("Model 1 loaded after weight modification")
            else:
                raise e
    else:
        logger.info(f"Model 1 checkpoint not found: {model1_path}")
        
    # Resume and load model 2
    model2_path = args.resume_from2
    logger.info(f'Model 2 resume from: {model2_path}')
    logger.info(f'Model 2 work dir: {args.work_dir2}')

    if model2_path and osp.exists(model2_path):
        map_location = 'cpu'
        ckpt = torch.load(model2_path, map_location=map_location)
        try:
            raw_model2.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)
            logger.info("Model 2 loaded successfully")
        except Exception as e:
            logger.info(f"Error loading model 2: {e}")
            if args.model2_type == "base":
                os.system(f"python modify_weight.py --work-dir {args.work_dir2} --epoch {args.epoch}")
                ckpt = torch.load(model2_path, map_location=map_location)
                raw_model2.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)
                logger.info("Model 2 loaded after weight modification")
            else:
                raise e
    else:
        logger.info(f"Model 2 checkpoint not found: {model2_path}")
    
    # Evaluation and Visualization settings
    print_freq = 1
    save_dir = common_save_dir
    os.makedirs(save_dir, exist_ok=True)

    model1.eval()
    model2.eval()

    # Initialize metrics if needed (make this optional)
    calculate_metrics = args.calculate_metrics
    miou_metric = None
    if calculate_metrics:
        try:
            from misc.metric_util import MeanIoU
            miou_metric = MeanIoU(
                list(range(1, 17)),
                17, #17,
                ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
                'vegetation'],
                True, 17, filter_minmax=False)
            logger.info('Metrics calculation initialized successfully')
        except Exception as e:
            logger.info(f'Failed to initialize metrics: {e}')
            calculate_metrics = False

    # Set parameters for Gaussian visualization based on model type
    draw_gaussian_params1 = {}
    if args.model1_type == "base":
        draw_gaussian_params1 = dict(
            scalar = 1.5,
            ignore_opa = False,
            filter_zsize = False
        )
    elif args.model1_type == "prob":
        draw_gaussian_params1 = dict(
            scalar = 2.0,
            ignore_opa = True,
            filter_zsize = True
        )
        
    draw_gaussian_params2 = {}
    if args.model2_type == "base":
        draw_gaussian_params2 = dict(
            scalar = 1.5,
            ignore_opa = False,
            filter_zsize = False
        )
    elif args.model2_type == "prob":
        draw_gaussian_params2 = dict(
            scalar = 2.0,
            ignore_opa = True,
            filter_zsize = True
        )

    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            # Copy the data for each model to avoid modifications affecting the other model
            data1 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            data2 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            
            for k in list(data1.keys()):
                if isinstance(data1[k], torch.Tensor):
                    data1[k] = data1[k].cuda()
            input_imgs1 = data1.pop('img')
            ori_imgs1 = data1.pop('ori_img')
            
            for k in list(data2.keys()):
                if isinstance(data2[k], torch.Tensor):
                    data2[k] = data2[k].cuda()
            input_imgs2 = data2.pop('img')
            ori_imgs2 = data2.pop('ori_img')
            
            # Save original input images (only need to do this once since inputs are the same)
            for i in range(ori_imgs1.shape[-1]):
                ori_img = ori_imgs1[0, ..., i].cpu().numpy()
                ori_img = ori_img[..., [2, 1, 0]]
                ori_img = Image.fromarray(ori_img.astype(np.uint8))
                ori_img.save(os.path.join(save_dir, f'{i_iter_val}_image_{i}.png'))
            
            # Run inference on both models
            result_dict1 = model1(imgs=input_imgs1, metas=data1)
            result_dict2 = model2(imgs=input_imgs2, metas=data2)
            
            # Process results from both models
            for idx, (pred1, pred2) in enumerate(zip(result_dict1['final_occ'], result_dict2['final_occ'])):
                pred_occ1 = pred1
                pred_occ2 = pred2
                gt_occ = result_dict1['sampled_label'][idx]  # Ground truth should be the same for both models
                occ_shape = [200, 200, 16]
                
                # Save top-down view of Gaussians if requested
                if args.vis_gaussian_topdown:
                    save_gaussian_topdown(
                        save_dir,
                        result_dict1['anchor_init'],
                        result_dict1['gaussians'],
                        f'val_{i_iter_val}_model1_topdown'
                    )
                    save_gaussian_topdown(
                        save_dir,
                        result_dict2['anchor_init'],
                        result_dict2['gaussians'],
                        f'val_{i_iter_val}_model2_topdown'
                    )
                
                # Save occupancy predictions if requested
                if args.vis_occ:
                    # Model 1 prediction
                    save_occ(
                        save_dir,
                        pred_occ1.reshape(1, *occ_shape),
                        f'val_{i_iter_val}_model1_pred',
                        True, 0, dataset=args.dataset)
                    
                    # Model 2 prediction
                    save_occ(
                        save_dir,
                        pred_occ2.reshape(1, *occ_shape),
                        f'val_{i_iter_val}_model2_pred',
                        True, 0, dataset=args.dataset)
                    
                    # Ground truth (only need to save once)
                    save_occ(
                        save_dir,
                        gt_occ.reshape(1, *occ_shape),
                        f'val_{i_iter_val}_gt',
                        True, 0, dataset=args.dataset)
                
                # Save Gaussian visualizations if requested
                if args.vis_gaussian:
                    save_gaussian(
                        save_dir,
                        result_dict1['gaussian'],
                        f'val_{i_iter_val}_model1_gaussian',
                        **draw_gaussian_params1)
                    
                    save_gaussian(
                        save_dir,
                        result_dict2['gaussian'],
                        f'val_{i_iter_val}_model2_gaussian',
                        **draw_gaussian_params2)
                
                # Calculate metrics for both models (if enabled)
                if calculate_metrics and miou_metric is not None:
                    try:
                        miou_metric._after_step(pred_occ1, gt_occ)
                        # Note: We're only calculating metrics for model 1 for simplicity
                    except Exception as e:
                        if i_iter_val == 0:  # Only log the error once
                            logger.info(f'Error calculating metrics: {e}')
                            calculate_metrics = False
            
            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Iter %5d'%(i_iter_val))
                    
    # Calculate and report metrics (if enabled)
    if calculate_metrics and miou_metric is not None:
        try:
            miou, iou2 = miou_metric._after_epoch()
            logger.info(f'Model 1 mIoU: {miou}, iou2: {iou2}')
            miou_metric.reset()
        except Exception as e:
            logger.info(f'Error finalizing metrics: {e}')
    
    logger.info('Visualization completed successfully!')


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Comparison visualization for two GaussianFormer models')
    # Config and work directory for model 1
    parser.add_argument('--py-config1', required=True, help='Python config file for model 1')
    parser.add_argument('--work-dir1', type=str, required=True, help='Work directory for model 1')
    parser.add_argument('--resume-from1', type=str, required=True, help='Checkpoint to resume from for model 1')
    parser.add_argument('--model1-type', type=str, default="base", choices=["base", "prob"], help='Type of model 1')
    
    # Config and work directory for model 2
    parser.add_argument('--py-config2', required=True, help='Python config file for model 2')
    parser.add_argument('--work-dir2', type=str, required=True, help='Work directory for model 2')
    parser.add_argument('--resume-from2', type=str, required=True, help='Checkpoint to resume from for model 2')
    parser.add_argument('--model2-type', type=str, default="base", choices=["base", "prob"], help='Type of model 2')
    
    # Common output directory
    parser.add_argument('--common-save-dir', type=str, required=True, help='Common directory to save comparison results')
    
    # Other settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vis-occ', action='store_true', default=False, help='Visualize occupancy maps')
    parser.add_argument('--vis-gaussian', action='store_true', default=False, help='Visualize gaussians')
    parser.add_argument('--vis_gaussian_topdown', action='store_true', default=False, help='Visualize gaussians in topdown view')
    parser.add_argument('--vis-index', type=int, nargs='+', default=[], help='Indices to visualize')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--vis_scene_index', type=int, default=-1, help='Scene index to visualize')
    parser.add_argument('--vis-scene', action='store_true', default=False, help='Visualize scene')
    parser.add_argument('--epoch', type=int, default=0, help='Epoch for weight modification if needed')
    parser.add_argument('--dataset', type=str, default='nusc', help='Dataset name')
    parser.add_argument('--calculate-metrics', action='store_true', default=False, help='Calculate metrics (may not work if MeanIoU class has changed)')
    
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
