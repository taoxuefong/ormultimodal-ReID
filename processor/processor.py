import logging
import time
import torch
import os
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    arguments2 = {}
    arguments2["num_epoch"] = num_epoch
    arguments2["iteration"] = 0

    arguments3 = {}
    arguments3["num_epoch"] = num_epoch
    arguments3["iteration"] = 0

    arguments4 = {}
    arguments4["num_epoch"] = num_epoch
    arguments4["iteration"] = 0

    logger = logging.getLogger("CLIP2ReID.train")
    logger.info('start training')

    loss_meter = AverageMeter()
    mcm_loss_meter = AverageMeter()
    mlm_loss_meter = AverageMeter()
    mcq_loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # 安全创建SummaryWriter
    try:
        tb_writer = SummaryWriter(log_dir=args.output_dir)
    except Exception as e:
        print(f"WARNING: Failed to create SummaryWriter: {e}")
        print("Continuing without tensorboard logging...")
        tb_writer = None

    best_ttop1 = 0.0
    best_stop1 = 0.0
    best_itop1 = 0.0
    best_ftop1 = 0.0
    best_loss = float('inf')  # 用于跟踪最佳损失值

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        mcm_loss_meter.reset()
        mlm_loss_meter.reset()
        mcq_loss_meter.reset()
        
        # 添加训练数据加载器信息
        logger.info(f"Starting Epoch {epoch}")
        logger.info(f"Train loader length: {len(train_loader)}")
        logger.info(f"Log period: {log_period}")
        logger.info(f"Batch size: {train_loader.batch_size if hasattr(train_loader, 'batch_size') else 'unknown'}")
        
        model.train()
        
        for n_iter, batch in enumerate(train_loader):
            try:
                # 添加批次信息（前几个批次）
                if n_iter < 3:
                    logger.info(f"Epoch {epoch}, Batch {n_iter}: batch keys = {list(batch.keys())}")
                    if 'vis_images' in batch:
                        logger.info(f"Epoch {epoch}, Batch {n_iter}: vis_images shape = {batch['vis_images'].shape}")
                    if 'pids' in batch:
                        logger.info(f"Epoch {epoch}, Batch {n_iter}: pids = {batch['pids']}")
                
                batch = {k: v.to(device) for k, v in batch.items()}

                ret = model(batch)
                
                # 检查返回的损失值
                loss_keys = [k for k in ret.keys() if "loss" in k]
                
                # 计算总损失
                if loss_keys:
                    total_loss = sum([ret[k] for k in loss_keys])
                else:
                    total_loss = torch.tensor(0.0, device=device)
                
                # 兼容不同的数据集键名
                batch_size = batch.get('images', batch.get('vis_images', torch.tensor(1))).shape[0]
                loss_meter.update(total_loss.item(), batch_size)
                acc_meter.update(ret.get('acc', 0), 1)
                
                # 更新各个损失分量
                # 使用模型实际返回的损失键名
                itc_loss = ret.get('itc_loss', 0)
                nir_loss = ret.get('nir_loss', 0)
                cp_loss = ret.get('cp_loss', 0)
                sk_loss = ret.get('sk_loss', 0)
                text_loss = ret.get('text_loss', 0)
                
                # 更新损失计量器
                mcm_loss_meter.update(itc_loss, batch_size)      # 使用itc_loss作为mcm_loss
                mlm_loss_meter.update(text_loss, batch_size)     # 使用text_loss作为mlm_loss
                mcq_loss_meter.update(nir_loss + cp_loss + sk_loss, batch_size)  # 使用多模态损失作为mcq_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                synchronize()

                # 每个batch都输出损失值
                print(f"Epoch {epoch}, Batch {n_iter}: Total Loss = {total_loss.item():.4f}, ITC = {itc_loss:.4f}, Text = {text_loss:.4f}, MultiModal = {nir_loss + cp_loss + sk_loss:.4f}")
                logger.info(f"Epoch {epoch}, Batch {n_iter}: Total Loss = {total_loss.item():.4f}, ITC = {itc_loss:.4f}, Text = {text_loss:.4f}, MultiModal = {nir_loss + cp_loss + sk_loss:.4f}")
                
                # 每10个batch输出一次详细统计
                if (n_iter + 1) % log_period == 0:
                    print(f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}] Loss: {loss_meter.avg:.4f}, itc_loss: {mcm_loss_meter.avg:.4f}, text_loss: {mlm_loss_meter.avg:.4f}, multimodal_loss: {mcq_loss_meter.avg:.4f}, Acc: {acc_meter.avg:.3f}, Base Lr: {scheduler.get_lr()[0]:.2e}")
                    logger.info(
                        f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}] Loss: {loss_meter.avg:.4f}, itc_loss: {mcm_loss_meter.avg:.4f}, text_loss: {mlm_loss_meter.avg:.4f}, multimodal_loss: {mcq_loss_meter.avg:.4f}, Acc: {acc_meter.avg:.3f}, Base Lr: {scheduler.get_lr()[0]:.2e}"
                    )
                
            except Exception as e:
                print(f"ERROR: Failed to process batch {n_iter}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 开始完整的训练流程
        
        # 安全添加tensorboard记录
        if tb_writer is not None:
            try:
                tb_writer.add_scalar('lr', scheduler.get_lr()[0] if hasattr(scheduler, 'get_lr') and len(scheduler.get_lr()) > 0 else args.lr, epoch)
                tb_writer.add_scalar('temperature', ret.get('temperature', 0.07), epoch)
                tb_writer.add_scalar('loss', loss_meter.avg, epoch)
                tb_writer.add_scalar('itc_loss', mcm_loss_meter.avg, epoch)
                tb_writer.add_scalar('text_loss', mlm_loss_meter.avg, epoch)
                tb_writer.add_scalar('multimodal_loss', mcq_loss_meter.avg, epoch)
                tb_writer.add_scalar('acc', acc_meter.avg, epoch)
            except Exception as e:
                print(f"WARNING: Failed to write tensorboard logs: {e}")

        # 更新学习率
        scheduler.step()
        
        # 记录训练统计
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        
        # 保存每个epoch的检查点
        if get_rank() == 0:
            # 每20个epoch保存一次检查点
            if epoch % 20 == 0:
                arguments["epoch"] = epoch
                checkpointer.save(f"model_epoch_{epoch}", **arguments)
                logger.info(f"Saved checkpoint for epoch {epoch}")
            
            # 保存最佳模型（基于当前epoch的损失）
            if epoch == 1 or loss_meter.avg < best_loss:
                best_loss = loss_meter.avg
                arguments["epoch"] = epoch
                checkpointer.save("model_best", **arguments)
                logger.info(f"Saved best model checkpoint (loss: {best_loss:.4f})")
        
        # 跳过评估（因为我们设置了eval_period=999999）
        # if epoch % eval_period == 0:
        #     ... (评估代码)
        
        continue
    
    # 训练完成后保存最终模型
    if get_rank() == 0:
        arguments["epoch"] = num_epoch
        checkpointer.save("model_final", **arguments)
        logger.info(f"Training completed! Final model saved.")
        
        # 保存训练完成信息
        logger.info(f"Final training statistics:")
        logger.info(f"  - Total epochs: {num_epoch}")
        logger.info(f"  - Final loss: {loss_meter.avg:.4f}")
        logger.info(f"  - Best loss: {best_loss:.4f}")
        logger.info(f"  - Model saved as: model_final")


def do_inference(args, model, test_img_loader, test_txt_loader, test_sketch_loader):

    logger = logging.getLogger("CLIP2ReID.test")
    logger.info("Enter inferencing")

    if args.dataset_name == 'PRCV':
        from utils.multimodal_evaluator import MultiModalEvaluator
        evaluator = MultiModalEvaluator(args, test_img_loader, test_txt_loader)
        results = evaluator.eval(model.eval())
        
        # Save ranking results
        output_path = os.path.join(args.output_dir, 'ranking_results.txt')
        evaluator.save_ranking_results(results, output_path)
        
        logger.info("Multi-modal evaluation completed")
        logger.info(f"Results saved to {output_path}")
    else:
        evaluator = Evaluator(args, test_img_loader, test_txt_loader, test_sketch_loader)
        ttop1, stop1, itop1 = evaluator.eval(model.eval())
        # top1 = evaluator.eval_by_proj(model.eval())

        # table = PrettyTable(["task", "R1", "R5", "R10", "mAP"])
        # table.float_format = '.4'
        # table.add_row(['t2i', cmc[0], cmc[4], cmc[9], mAP])
        # logger.info("Validation Results: ")
        # logger.info('\n' + str(table))
