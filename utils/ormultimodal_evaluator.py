import torch
import numpy as np
import logging
from typing import Dict, List, Tuple
from .metrics import eval_func
import os
import json
import time


class PRCVEvaluator:
    """专门用于PRCV数据集的简单评估器"""
    
    def __init__(self, args, gallery_loader, query_loader):
        self.args = args
        self.gallery_loader = gallery_loader
        self.query_loader = query_loader
        self.logger = logging.getLogger("CLIP2ReID.evaluator")
        
    def eval(self, model):
        """评估模型，输出检索结果文件"""
        model.eval()
        
        # 提取gallery特征
        gallery_features, gallery_pids, gallery_camids = self._extract_gallery_features(model)
        
        # 提取query特征
        query_features, query_pids, query_camids = self._extract_query_features(model)
        
        # 计算距离矩阵
        distmat = self._compute_distance_matrix(query_features, gallery_features)
        
        # 生成检索结果
        ranking_results = self._generate_ranking_results(distmat)
        
        # 保存结果到文件
        output_file = os.path.join(self.args.output_dir, 'val_ranking_results.csv')
        self._save_ranking_results(ranking_results, output_file)
        
        self.logger.info(f"Retrieval results saved to: {output_file}")
        
        # 返回空的评估指标（兼容训练流程）
        cmc = np.zeros(10)
        mAP = 0.0
        indices = np.empty((0, 0))
        
        return cmc, mAP, indices
    
    def _extract_gallery_features(self, model):
        model.eval()
        device = next(model.parameters()).device
        features = []
        pids = []
        camids = []
        
        print(f"DEBUG: Starting gallery feature extraction...")
        print(f"DEBUG: Gallery loader type: {type(self.gallery_loader)}")
        
        # 计算总批次数
        total_batches = len(self.gallery_loader)
        print(f"DEBUG: Total gallery batches to process: {total_batches}")
        
        with torch.no_grad():
            batch_count = 0
            start_time = time.time()
            
            for batch in self.gallery_loader:
                batch_count += 1
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # 每10个批次输出一次进度
                if batch_count % 10 == 0:
                    avg_time_per_batch = elapsed_time / batch_count
                    remaining_batches = total_batches - batch_count
                    estimated_remaining_time = remaining_batches * avg_time_per_batch
                    print(f"DEBUG: Progress: {batch_count}/{total_batches} batches processed")
                    print(f"DEBUG: Elapsed time: {elapsed_time:.2f}s, Avg time per batch: {avg_time_per_batch:.2f}s")
                    print(f"DEBUG: Estimated remaining time: {estimated_remaining_time:.2f}s")
                
                # 检查是否超时（如果单个批次处理超过30秒，认为卡住了）
                if batch_count > 1 and (current_time - start_time) / batch_count > 30:
                    print(f"WARNING: Batch {batch_count} taking too long, possible hang detected!")
                    print(f"DEBUG: Current batch processing time: {current_time - start_time:.2f}s")
                
                # print(f"DEBUG: Processing gallery batch {batch_count}/{total_batches}")
                
                # gallery_loader 返回 (pid, img, image_id) 格式
                if isinstance(batch, (list, tuple)):
                    if len(batch) >= 3:
                        pid = batch[0]
                        img = batch[1]
                        image_id = batch[2]
                    elif len(batch) == 2:
                        pid = batch[0]
                        img = batch[1]
                        image_id = 0
                    else:
                        # print(f"DEBUG: Skipping batch with insufficient elements: {len(batch)}")
                        continue
                else:
                    # print(f"DEBUG: Skipping non-list/tuple batch: {type(batch)}")
                    continue
                
                # print(f"DEBUG: Batch {batch_count} - pid: {pid}, img shape: {img.shape if torch.is_tensor(img) else 'not tensor'}")
                
                # 将图像移动到 GPU
                if torch.is_tensor(img):
                    img = img.to(device)
                    batch_data = {'vis_images': img}
                else:
                    # print(f"DEBUG: Skipping non-tensor image in batch {batch_count}")
                    continue
                
                # 提取特征
                # print(f"DEBUG: Extracting features for batch {batch_count}...")
                try:
                    output = model(batch_data)
                    if isinstance(output, dict):
                        feat = output.get('gallery_features', None)
                        if feat is None:
                            # print(f"DEBUG: No gallery_features found in output for batch {batch_count}")
                            continue
                    else:
                        feat = output
                except Exception as e:
                    print(f"ERROR: Feature extraction failed for batch {batch_count}: {e}")
                    continue
                
                # print(f"DEBUG: Batch {batch_count} - extracted feature shape: {feat.shape}")
                
                # 确保特征是二维张量
                if feat.dim() > 2:
                    feat = feat.view(feat.size(0), -1)  # 展平为 (batch_size, feature_dim)
                elif feat.dim() == 1:
                    feat = feat.unsqueeze(0)  # 转换为 (1, feature_dim)
                
                batch_size = feat.size(0)
                features.extend([feat[i].cpu() for i in range(batch_size)])
                
                # 正确处理批次张量中的 pid
                if isinstance(pid, torch.Tensor):
                    if pid.numel() == 1:
                        pid = pid.item()
                        pids.extend([int(pid)] * batch_size)
                    else:
                        # 如果 pid 是批次张量，使用其中的值
                        pid_list = pid.cpu().numpy().tolist()
                        pids.extend([int(p) for p in pid_list])
                elif isinstance(pid, (list, tuple)):
                    pids.extend([int(p) for p in pid])
                else:
                    pids.extend([int(pid)] * batch_size)
                
                # 设置默认的 camid
                camid = 0
                camids.extend([camid] * batch_size)
                
                # print(f"DEBUG: Batch {batch_count} completed - features: {len(features)}, pids: {len(pids)}")
        
        total_time = time.time() - start_time
        print(f"DEBUG: Gallery feature extraction completed. Total batches: {batch_count}")
        print(f"DEBUG: Total features collected: {len(features)}")
        print(f"DEBUG: Total time: {total_time:.2f}s, Average time per batch: {total_time/batch_count:.2f}s")
        
        # 修复特征连接问题：确保所有特征都是二维的
        if len(features) > 0:
            # 检查第一个特征的形状
            first_shape = features[0].shape
            if len(first_shape) == 1:
                # 如果是一维特征，转换为二维
                features = [f.unsqueeze(0) if f.dim() == 1 else f for f in features]
                # print(f"DEBUG: Gallery - Converted features to 2D, first shape: {features[0].shape}")
        
        # print(f"DEBUG: Concatenating {len(features)} gallery features...")
        features = torch.cat(features, dim=0) if features else torch.empty(0, device='cpu')
        print(f"DEBUG: Gallery - After concatenation: {features.shape}")
        
        # 确保最终特征是二维的
        if features.dim() == 1:
            # 如果是一维，尝试重塑为二维
            if hasattr(model, 'embed_dim'):
                expected_dim = model.embed_dim
                if features.numel() % expected_dim == 0:
                    batch_size = features.numel() // expected_dim
                    features = features.view(batch_size, expected_dim)
                    print(f"DEBUG: Gallery - Reshaped 1D features to 2D: {features.shape}")
        
        return features, np.array(pids, dtype=np.int32), np.array(camids, dtype=np.int32)
    
    def _extract_query_features(self, model):
        model.eval()
        device = next(model.parameters()).device
        features = []
        pids = []
        camids = []
        
        with torch.no_grad():
            for batch in self.query_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]  # 如果是列表或元组，取第一个元素
                query_idx = batch['query_idx']
                modalities = batch['modalities']
                content = batch.get('content', [])
                
                # 将 batch 移动到 GPU
                batch_data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # 提取特征
                output = model(batch_data)
                if isinstance(output, dict):
                    feat = output.get('query_features', None)  # 使用正确的键名 'query_features'
                    if feat is None:
                        continue
                else:
                    feat = output
                
                # 调试信息：打印原始特征形状
                # print(f"DEBUG: Original feat shape: {feat.shape}")
                # print(f"DEBUG: feat.dim(): {feat.dim()}")
                
                # 确保特征是二维张量，保持正确的特征维度
                if feat.dim() > 2:
                    # 如果特征维度大于2，展平为 (batch_size, feature_dim)
                    feat = feat.view(feat.size(0), -1)
                    # print(f"DEBUG: After flattening: {feat.shape}")
                elif feat.dim() == 1:
                    # 如果特征维度为1，转换为 (1, feature_dim)
                    feat = feat.unsqueeze(0)
                    # print(f"DEBUG: After unsqueeze: {feat.shape}")
                
                # 检查特征维度是否合理
                if feat.size(-1) > 10000:  # 如果特征维度过大，可能是展平错误
                    print(f"WARNING: Feature dimension too large: {feat.size()}")
                    # 尝试重新获取正确的特征维度
                    if hasattr(model, 'embed_dim'):
                        expected_dim = model.embed_dim
                        # print(f"DEBUG: Expected dimension: {expected_dim}")
                        if feat.size(0) * expected_dim == feat.numel():
                            # 重新整形为正确的维度
                            feat = feat.view(feat.size(0), expected_dim)
                            print(f"Fixed feature dimension to: {feat.size()}")
                        # else: # Debug print
                            # print(f"DEBUG: Cannot reshape: {feat.size(0)} * {expected_dim} != {feat.numel()}")
                
                batch_size = feat.size(0) if feat.dim() > 1 else 1
                # print(f"DEBUG: Final feat shape before storage: {feat.shape}")
                features.extend([feat[i].cpu() for i in range(batch_size)] if batch_size > 1 else [feat.cpu()])
                
                # Debug prints for features list
                # print(f"DEBUG: Features list length after extend: {len(features)}")
                # if len(features) > 0:
                #     print(f"DEBUG: Last stored feature shape: {features[-1].shape}")
                #     print(f"DEBUG: Last stored feature dim: {features[-1].dim()}")
                
                # 提取 pid 和 camid
                if isinstance(query_idx, torch.Tensor) and query_idx.dim() > 0:
                    batch_pids = query_idx.tolist()  # 转换为列表
                else:
                    batch_pids = [query_idx] if not isinstance(query_idx, list) else query_idx
                
                for idx in range(batch_size):
                    pid = None
                    camid = None
                    if idx < len(content) and isinstance(content[idx], list):
                        for modality, cont in zip(modalities, content[idx] if isinstance(content[idx], list) else []):
                            if modality in ['nir', 'cp', 'sk'] and isinstance(cont, str) and cont.endswith(('.jpg', '.jpeg', '.png')):
                                # 从图片名称中提取 pid 和 camid
                                parts = cont.split('_')
                                if parts:
                                    pid_str = parts[0]  # 例如 0001
                                    pid = int(pid_str) if pid_str.isdigit() else None
                                    
                                    if modality == 'nir' and len(parts) > 2:
                                        # NIR 模态：0001_llcm_0001_c01_s185535_f7000_nir.jpg
                                        camid_str = parts[2]  # 例如 c01
                                        if camid_str.startswith('c') and camid_str[1:].isdigit():
                                            camid = int(camid_str[1:])  # c01 -> 1, c02 -> 2, c03 -> 3
                                    
                                    elif modality in ['cp', 'sk'] and len(parts) > 3:
                                        # Sketch/CP 模态：0001_llcm_0001_back_0_sketch.jpg
                                        angle = parts[3].lower()  # 例如 back
                                        if angle == 'back':
                                            camid = 1
                                        elif angle == 'front':
                                            camid = 2
                                        elif angle == 'side':
                                            camid = 3
                                    
                                    # 如果成功提取了 pid 和 camid，跳出循环
                                    if pid is not None and camid is not None:
                                        break
                    
                    # 如果从文件名中无法提取，使用默认值
                    if pid is None:
                        pid = batch_pids[idx] if idx < len(batch_pids) else 0
                    if camid is None:
                        camid = 0  # 默认摄像头ID
                    
                    pids.append(int(pid))
                    camids.append(int(camid))
        
        # 调试信息：检查连接前的特征列表
        # print(f"DEBUG: Total features to concatenate: {len(features)}")
        # if len(features) > 0:
        #     print(f"DEBUG: First feature shape: {features[0].shape}")
        #     print(f"DEBUG: Last feature shape: {features[-1].shape}")
        #     # 检查所有特征的形状是否一致
        #     shapes = [f.shape for f in features]
        #     print(f"DEBUG: All feature shapes: {shapes}")
        
        # 修复特征连接问题：确保所有特征都是二维的
        if len(features) > 0:
            # 检查第一个特征的形状
            first_shape = features[0].shape
            if len(first_shape) == 1:
                # 如果是一维特征，转换为二维
                features = [f.unsqueeze(0) if f.dim() == 1 else f for f in features]
                # print(f"DEBUG: Converted features to 2D, first shape: {features[0].shape}")
        
        features = torch.cat(features, dim=0) if features else torch.empty(0, device='cpu')
        print(f"DEBUG: After concatenation: {features.shape}")
        
        # 确保最终特征是二维的
        if features.dim() == 1:
            # 如果是一维，尝试重塑为二维
            if hasattr(model, 'embed_dim'):
                expected_dim = model.embed_dim
                if features.numel() % expected_dim == 0:
                    batch_size = features.numel() // expected_dim
                    features = features.view(batch_size, expected_dim)
                    # print(f"DEBUG: Reshaped 1D features to 2D: {features.shape}")
        
        return features, np.array(pids, dtype=np.int32), np.array(camids, dtype=np.int32)
    
    def _compute_distance_matrix(self, query_features, gallery_features):
        """分批计算距离矩阵，避免内存溢出"""
        print(f"DEBUG: Starting distance matrix computation...")
        print(f"DEBUG: Query features: {query_features.shape}, Gallery features: {gallery_features.shape}")
        
        # 确保特征是二维张量
        if query_features.dim() == 1:
            query_features = query_features.unsqueeze(0)
        if gallery_features.dim() == 1:
            gallery_features = gallery_features.unsqueeze(0)
        
        # 检查特征维度是否一致
        if query_features.size(-1) != gallery_features.size(-1):
            raise ValueError(f"Feature dimensions do not match: query_features {query_features.size()}, gallery_features {gallery_features.size()}")
        
        # 归一化特征
        query_features = torch.nn.functional.normalize(query_features, dim=-1)
        gallery_features = torch.nn.functional.normalize(gallery_features, dim=-1)
        
        # 分批计算距离矩阵
        num_queries = query_features.shape[0]
        num_gallery = gallery_features.shape[0]
        batch_size = 1000  # 每批处理1000个查询
        
        print(f"DEBUG: Processing {num_queries} queries in batches of {batch_size}")
        
        # 初始化距离矩阵
        distmat = np.zeros((num_queries, num_gallery), dtype=np.float32)
        
        start_time = time.time()
        for i in range(0, num_queries, batch_size):
            end_idx = min(i + batch_size, num_queries)
            batch_queries = query_features[i:end_idx]
            
            # 计算当前批次的距离
            with torch.no_grad():
                # 计算余弦相似度
                batch_sim = torch.matmul(batch_queries, gallery_features.t())
                # 转换为距离 (1 - 相似度)
                batch_dist = 1 - batch_sim
                distmat[i:end_idx] = batch_dist.cpu().numpy()
            
            # 显示进度
            if i % 5000 == 0 or end_idx == num_queries:
                elapsed = time.time() - start_time
                progress = end_idx / num_queries * 100
                print(f"DEBUG: Distance computation progress: {end_idx}/{num_queries} ({progress:.1f}%) - Elapsed: {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        print(f"DEBUG: Distance matrix computation completed in {total_time:.1f}s")
        print(f"DEBUG: Distance matrix shape: {distmat.shape}")
        
        return distmat
    
    def _generate_ranking_results(self, distmat):
        """生成检索结果，每个查询返回前100个样本的索引"""
        print(f"DEBUG: Starting to generate ranking results...")
        print(f"DEBUG: Distance matrix shape: {distmat.shape}")
        
        # 批量读取所有query_type和query_idx，避免重复文件读取
        print(f"DEBUG: Loading query types and indices from val_queries.json...")
        all_query_info = self._get_all_query_info()
        print(f"DEBUG: Loaded {len(all_query_info)} query info")
        
        ranking_results = []
        num_queries = distmat.shape[0]
        
        start_time = time.time()
        for query_idx in range(num_queries):
            # 获取当前查询的距离
            distances = distmat[query_idx, :]
            
            # 使用numpy的argpartition优化排序（只取前100个）
            top_100_indices = np.argpartition(distances, 100)[:100]
            # 对前100个进行排序
            top_100_sorted = top_100_indices[np.argsort(distances[top_100_indices])]
            
            # 转换为列表
            top_100_indices = top_100_sorted.tolist()
            
            # 从预加载的query_info中获取
            query_info = all_query_info.get(query_idx, {'query_idx': query_idx, 'query_type': f"unknown_{query_idx}"})
            original_query_idx = query_info['query_idx']
            query_type = query_info['query_type']
            
            ranking_results.append({
                'query_idx': original_query_idx,  # 使用原始的query_idx
                'query_type': query_type,
                'ranking_list_idx': top_100_indices
            })
            
            # 每1000个查询显示一次进度
            if (query_idx + 1) % 1000 == 0 or query_idx == num_queries - 1:
                elapsed = time.time() - start_time
                progress = (query_idx + 1) / num_queries * 100
                avg_time_per_query = elapsed / (query_idx + 1)
                remaining_queries = num_queries - (query_idx + 1)
                estimated_remaining_time = remaining_queries * avg_time_per_query
                print(f"DEBUG: Ranking generation progress: {query_idx + 1}/{num_queries} ({progress:.1f}%) - Elapsed: {elapsed:.1f}s")
                print(f"DEBUG: Avg time per query: {avg_time_per_query:.3f}s, Estimated remaining: {estimated_remaining_time:.1f}s")
        
        total_time = time.time() - start_time
        print(f"DEBUG: Ranking results generation completed in {total_time:.1f}s")
        print(f"DEBUG: Generated {len(ranking_results)} ranking results")
        
        return ranking_results
    
    def _get_all_query_info(self):
        """批量读取所有query信息，包括query_idx和query_type，避免重复文件读取"""
        try:
            val_queries_path = os.path.join(self.args.root_dir, 'val', 'val_queries.json')
            with open(val_queries_path, 'r', encoding='utf-8') as f:
                queries = json.load(f)
            
            # 创建索引到query信息的映射
            # 注意：DataLoader会按照val_queries.json的顺序加载数据
            # 所以索引i对应第i个查询
            query_info_map = {}
            for i, query in enumerate(queries):
                query_info_map[i] = {
                    'query_idx': query['query_idx'],  # 原始的query_idx
                    'query_type': query['query_type']
                }
            
            print(f"DEBUG: First few queries: {list(query_info_map.items())[:5]}")
            print(f"DEBUG: Last few queries: {list(query_info_map.items())[-5:]}")
            
            # 查找特定query_idx的位置
            target_query_idx = 5100
            for i, query in enumerate(queries):
                if query['query_idx'] == target_query_idx:
                    print(f"DEBUG: query_idx {target_query_idx} found at index {i}: query_type={query['query_type']}")
                    break
            
            return query_info_map
        except Exception as e:
            print(f"WARNING: Failed to load query info: {e}")
            # 返回默认映射
            return {i: {'query_idx': i, 'query_type': f"unknown_{i}"} for i in range(78526)}
    
    def _save_ranking_results(self, ranking_results, output_file):
        """保存检索结果到CSV文件"""
        print(f"DEBUG: Starting to save ranking results to {output_file}")
        print(f"DEBUG: Total results to save: {len(ranking_results)}")
        
        start_time = time.time()
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 修改文件扩展名为.csv
        csv_file = output_file.replace('.txt', '.csv')
        
        import csv
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            # 创建CSV writer
            fieldnames = ['query_idx', 'query_type', 'ranking_list_idx']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # 写入表头
            writer.writeheader()
            
            # 批量写入结果
            batch_size = 1000
            for i in range(0, len(ranking_results), batch_size):
                batch = ranking_results[i:i + batch_size]
                
                # 格式化批次数据
                for result in batch:
                    # query_idx已经是原始值，不需要加1
                    query_idx = result['query_idx']
                    # 格式化ranking_list_idx为字符串
                    ranking_str = '[' + ','.join(map(str, result['ranking_list_idx'])) + ']'
                    
                    # 写入CSV行
                    writer.writerow({
                        'query_idx': query_idx,
                        'query_type': result['query_type'],
                        'ranking_list_idx': ranking_str
                    })
                
                # 显示进度
                if i % 5000 == 0 or i + batch_size >= len(ranking_results):
                    elapsed = time.time() - start_time
                    progress = min(i + batch_size, len(ranking_results)) / len(ranking_results) * 100
                    print(f"DEBUG: File saving progress: {min(i + batch_size, len(ranking_results))}/{len(ranking_results)} ({progress:.1f}%) - Elapsed: {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        print(f"DEBUG: Ranking results saved to {csv_file} in {total_time:.1f}s")
        print(f"DEBUG: File size: {os.path.getsize(csv_file) / 1024 / 1024:.2f} MB")
