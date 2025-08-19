import os
import os.path as op
import torch
import numpy as np
import random
import time

from datasets import build_dataloader
from processor.processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize

# 设置多进程共享策略为 file_system 以避免打开过多文件
torch.multiprocessing.set_sharing_strategy('file_system')


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    args = get_args()
    set_seed(1+get_rank())
    name = args.name

    # Set dataset to PRCV
    args.dataset_name = 'PRCV'
    args.root_dir = '/data/taoxuefeng/PRCV'
    
    # Set output directory to user's home directory or current directory
    if not hasattr(args, 'output_dir') or not args.output_dir:
        args.output_dir = './outputs'

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    
    # Create output directory with timestamp
    output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}')
    
    # Ensure the directory exists and is writable
    try:
        os.makedirs(output_dir, exist_ok=True)
        args.output_dir = output_dir
    except PermissionError:
        # Fallback to current directory if permission denied
        fallback_dir = op.join('./outputs', args.dataset_name, f'{cur_time}_{name}')
        os.makedirs(fallback_dir, exist_ok=True)
        args.output_dir = fallback_dir
        print(f"Warning: Using fallback output directory: {fallback_dir}")
    
    logger = setup_logger('MultiModalReID', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    logger.info("Training multi-modal ReID model for PRCV dataset")

    save_train_configs(args.output_dir, args)

    # Get dataloaders
    train_loader, val_img_loader, val_txt_loader, val_sketch_loader, num_classes = build_dataloader(args)
    
    # Build model
    model = build_model(args, num_classes)
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
    
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    
    # For PRCV, we use a custom evaluator
    if args.dataset_name == 'PRCV':
        from utils.prcv_evaluator import PRCVEvaluator
        evaluator = PRCVEvaluator(args, val_img_loader, val_txt_loader)
    else:
        evaluator = Evaluator(args, val_img_loader, val_txt_loader, val_sketch_loader)

    start_epoch = 1

    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']

    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)
