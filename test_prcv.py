from prettytable import PrettyTable
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.parallel
import numpy as np
import time
import os.path as op

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
    
    # 添加缺失的必要参数
    if not hasattr(args, 'num_workers'):
        args.num_workers = 4
    if not hasattr(args, 'batch_size'):
        args.batch_size = 32
    if not hasattr(args, 'test_batch_size'):
        args.test_batch_size = 512
    if not hasattr(args, 'img_size'):
        args.img_size = (384, 128)
    if not hasattr(args, 'text_length'):
        args.text_length = 77
    if not hasattr(args, 'sampler'):
        args.sampler = 'random'
    if not hasattr(args, 'num_instance'):
        args.num_instance = 4
    if not hasattr(args, 'val_dataset'):
        args.val_dataset = 'test'
    if not hasattr(args, 'img_aug'):
        args.img_aug = False
    if not hasattr(args, 'nlp_aug'):
        args.nlp_aug = False
    if not hasattr(args, 'MCQ'):
        args.MCQ = False
    if not hasattr(args, 'MCM'):
        args.MCM = False
    if not hasattr(args, 'MLM'):
        args.MLM = False
    if not hasattr(args, 'MSM'):
        args.MSM = False
    if not hasattr(args, 'MCQMLM'):
        args.MCQMLM = False
    if not hasattr(args, 'MSMMLM'):
        args.MSMMLM = False
    if not hasattr(args, 'distributed'):
        args.distributed = False
    if not hasattr(args, 'local_rank'):
        args.local_rank = 0
    if not hasattr(args, 'world_size'):
        args.world_size = 1
    
    # 添加模型构建相关的参数
    if not hasattr(args, 'pretrain_choice'):
        args.pretrain_choice = 'ViT-B/16'
    if not hasattr(args, 'stride_size'):
        args.stride_size = 16
    if not hasattr(args, 'temperature'):
        args.temperature = 0.07
    if not hasattr(args, 'use_imageid'):
        args.use_imageid = False
    if not hasattr(args, 'vocab_size'):
        args.vocab_size = 49408
    if not hasattr(args, 'pa'):
        args.pa = 0.1
    if not hasattr(args, 'power'):
        args.power = 0.9
    if not hasattr(args, 'target_lr'):
        args.target_lr = 1e-08
    if not hasattr(args, 'warmup_epochs'):
        args.warmup_epochs = 5
    if not hasattr(args, 'warmup_factor'):
        args.warmup_factor = 0.1
    if not hasattr(args, 'warmup_method'):
        args.warmup_method = 'linear'
    if not hasattr(args, 'weight_decay'):
        args.weight_decay = 4e-05
    if not hasattr(args, 'weight_decay_bias'):
        args.weight_decay_bias = 0.0
    if not hasattr(args, 'lr'):
        args.lr = 0.0001
    if not hasattr(args, 'num_epoch'):
        args.num_epoch = 100
    if not hasattr(args, 'log_period'):
        args.log_period = 10
    if not hasattr(args, 'eval_period'):
        args.eval_period = 10
    if not hasattr(args, 'save_period'):
        args.save_period = 20
    if not hasattr(args, 'resume'):
        args.resume = False
    if not hasattr(args, 'resume_ckpt_file'):
        args.resume_ckpt_file = ''
    if not hasattr(args, 'name'):
        args.name = 'baseline'
    if not hasattr(args, 'only_fusion_loss'):
        args.only_fusion_loss = False
    if not hasattr(args, 'only_sketch'):
        args.only_sketch = False
    if not hasattr(args, 'only_text'):
        args.only_text = False
    if not hasattr(args, 'optimizer'):
        args.optimizer = 'Adam'
    
    # Set output directory
    if not args.output_dir:
        args.output_dir = './outputs'
    
    # Ensure output directory exists and is writable
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except PermissionError:
        # Fallback to current directory if permission denied
        fallback_dir = './outputs'
        os.makedirs(fallback_dir, exist_ok=True)
        args.output_dir = fallback_dir
        print(f"Warning: Using fallback output directory: {fallback_dir}")
    
    logger = setup_logger('MultiModalReID', save_dir=args.output_dir, if_train=args.training)
    logger.info("Testing multi-modal ReID model for PRCV dataset")
    logger.info(args)
    
    device = "cuda"
    
    # Build dataloaders
    test_img_loader, test_txt_loader, test_sketch_loader = build_dataloader(args)
    
    # 调试：查看数据加载器的数据格式
    logger.info("Data loaders built successfully")
    logger.info(f"Test img loader: {len(test_img_loader)} batches")
    logger.info(f"Test txt loader: {len(test_txt_loader)} batches")
    if test_sketch_loader is not None:
        logger.info(f"Test sketch loader: {len(test_sketch_loader)} batches")
    else:
        logger.info("Test sketch loader: None (not available)")
    
    # 检查第一个batch的数据格式
    if len(test_img_loader) > 0:
        first_batch = next(iter(test_img_loader))
        logger.info(f"First gallery batch type: {type(first_batch)}")
        if isinstance(first_batch, (list, tuple)):
            logger.info(f"First gallery batch length: {len(first_batch)}")
            if len(first_batch) >= 3:
                logger.info(f"Gallery batch format: (pids, images, image_ids)")
                logger.info(f"Gallery pids shape: {first_batch[0].shape if hasattr(first_batch[0], 'shape') else 'N/A'}")
                logger.info(f"Gallery images shape: {first_batch[1].shape if hasattr(first_batch[1], 'shape') else 'N/A'}")
        else:
            logger.info(f"First gallery batch keys: {list(first_batch.keys())}")
            if 'vis_images' in first_batch:
                logger.info(f"Gallery images shape: {first_batch['vis_images'].shape}")
    
    if len(test_txt_loader) > 0:
        first_query_batch = next(iter(test_txt_loader))
        logger.info(f"First query batch type: {type(first_query_batch)}")
        if isinstance(first_query_batch, (list, tuple)):
            logger.info(f"First query batch length: {len(first_query_batch)}")
            if len(first_query_batch) >= 2:
                logger.info(f"Query batch format: (text_tokens, query_type, query_idx)")
                logger.info(f"Text tokens shape: {first_query_batch[0].shape if hasattr(first_query_batch[0], 'shape') else 'N/A'}")
        else:
            logger.info(f"First query batch keys: {list(first_query_batch.keys())}")
            if 'text_tokens' in first_query_batch:
                logger.info(f"Text tokens shape: {first_query_batch['text_tokens'].shape}")
    
    # Build model
    model = build_model(args)
    model.to(device)
    
    # 设置模型为评估模式
    model.eval()
    
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
    
    # 确保模型在评估模式
    model.eval()
    
    logger.info("Model loaded and set to evaluation mode")
    logger.info(f"Model type: {type(model)}")
    
    # 使用PRCVEvaluator进行推理
    if args.dataset_name == 'PRCV':
        from utils.prcv_evaluator import PRCVEvaluator
        evaluator = PRCVEvaluator(args, test_img_loader, test_txt_loader)
        logger.info("Using PRCVEvaluator for PRCV dataset")
        
        # 运行推理
        results = evaluator.eval(model.eval())
        logger.info("Inference completed successfully!")
        
        # PRCVEvaluator已经在eval方法中自动保存了结果
        # 结果保存在 args.output_dir/val_ranking_results.txt
        logger.info(f"Results have been automatically saved by PRCVEvaluator")
    else:
        # 对于其他数据集，使用原来的do_inference
        do_inference(args, model, test_img_loader, test_txt_loader, test_sketch_loader)
    
    logger.info("Testing completed!")
