import os
import os.path as op
import torch
import torch.nn.parallel
import numpy as np
import time

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Modal Person Re-identification Testing")
    parser.add_argument("--config_file", default='', help="Path to config file")
    parser.add_argument("--checkpoint", default='', help="Path to checkpoint file")
    parser.add_argument("--output_dir", default='./outputs', help="Output directory")
    args = parser.parse_args()
    
    # Load config if provided
    if args.config_file:
        args = load_train_configs(args.config_file)
    
    # Set dataset to PRCV
    args.dataset_name = 'PRCV'
    args.root_dir = '/data/taoxuefeng/PRCV'
    args.training = False
    
    # Set output directory to current directory to avoid permission issues
    if not args.output_dir or args.output_dir.startswith('/data1'):
        args.output_dir = './outputs'
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Output directory: {args.output_dir}")
    
    logger = setup_logger('MultiModalReID', save_dir=args.output_dir, if_train=args.training)
    logger.info("Testing multi-modal ReID model for PRCV dataset")
    logger.info(args)
    
    device = "cuda"
    
    # Build dataloaders
    test_img_loader, test_txt_loader, test_sketch_loader = build_dataloader(args)
    
    # Build model
    model = build_model(args)
    model.to(device)
    
    # Load checkpoint
    checkpointer = Checkpointer(model)
    if args.checkpoint:
        checkpointer.load(f=args.checkpoint)
        logger.info(f"Loaded checkpoint from {args.checkpoint}")
    else:
        # Try to load from output directory
        checkpoint_path = op.join(args.output_dir, 'multimodal_best.pth')
        if op.exists(checkpoint_path):
            checkpointer.load(f=checkpoint_path)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            logger.warning("No checkpoint found, using random weights")
    
    # Run inference
    do_inference(args, model, test_img_loader, test_txt_loader, test_sketch_loader)
    
    logger.info("Testing completed!")
