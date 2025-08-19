import torch
import numpy as np
import logging
from typing import Dict, List, Tuple
from .metrics import eval_func


class MultiModalEvaluator:
    """Evaluator for multi-modal person re-identification"""
    
    def __init__(self, args, gallery_loader, query_loader):
        self.args = args
        self.gallery_loader = gallery_loader
        self.query_loader = query_loader
        self.logger = logging.getLogger("CLIP2ReID.evaluator")
        
    def eval(self, model):
        """Evaluate the model"""
        model.eval()
        
        # Extract gallery features
        gallery_features, gallery_pids, gallery_camids = self._extract_gallery_features(model)
        
        # Extract query features for each query type
        query_results = self._extract_query_features(model)
        
        # Evaluate each query type
        results = {}
        for query_type, query_data in query_results.items():
            query_features = query_data['features']
            query_pids = query_data['pids']
            query_camids = query_data['camids']
            
            # Compute distance matrix
            distmat = self._compute_distance_matrix(query_features, gallery_features)
            
            # Evaluate
            cmc, mAP, indices = self._evaluate_rank(distmat, query_pids, gallery_pids, 
                                                   query_camids, gallery_camids)
            
            results[query_type] = {
                'cmc': cmc,
                'mAP': mAP,
                'indices': indices,
                'distmat': distmat
            }
            
            self.logger.info(f"{query_type} - R1: {cmc[0]:.2f}, R5: {cmc[4]:.2f}, "
                           f"R10: {cmc[9]:.2f}, mAP: {mAP:.2f}")
        
        return results
    
    def _extract_gallery_features(self, model):
        """Extract features from gallery images"""
        features = []
        pids = []
        camids = []
        
        with torch.no_grad():
            for batch in self.gallery_loader:
                # 确保batch是字典格式
                if isinstance(batch, list):
                    # 如果是列表，转换为字典
                    batch = {
                        'images': batch[0],  # 假设第一个元素是图像
                        'image_pids': batch[1] if len(batch) > 1 else torch.zeros(batch[0].shape[0]),
                        'image_camids': batch[2] if len(batch) > 2 else torch.zeros(batch[0].shape[0])
                    }
                
                # 将数据移到GPU
                batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
                
                ret = model(batch)
                feat = ret['gallery_features']
                features.append(feat.cpu())
                pids.extend(batch['image_pids'].cpu().numpy())
                camids.extend(batch.get('image_camids', [0] * len(batch['image_pids'])).cpu().numpy())
        
        features = torch.cat(features, dim=0)
        return features, np.array(pids), np.array(camids)
    
    def _extract_query_features(self, model):
        """Extract features from queries, grouped by query type"""
        query_results = {}
        
        with torch.no_grad():
            for batch in self.query_loader:
                # 确保batch是字典格式
                if isinstance(batch, list):
                    # 如果是列表，转换为字典
                    batch = {
                        'nir_images': batch[0] if len(batch) > 0 else None,
                        'cp_images': batch[1] if len(batch) > 1 else None,
                        'sk_images': batch[2] if len(batch) > 2 else None,
                        'text_tokens': batch[3] if len(batch) > 3 else None,
                        'query_type': 'unknown',
                        'query_idx': 0
                    }
                
                # 将数据移到GPU
                batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
                
                ret = model(batch)
                feat = ret['query_features']
                
                # Get query type - handle both list and single value
                if isinstance(batch['query_type'], list):
                    query_types = batch['query_type']
                else:
                    query_types = [batch['query_type']]
                
                # Get query indices
                query_indices = batch.get('query_idx', list(range(len(query_types))))
                if not isinstance(query_indices, list):
                    query_indices = [query_indices]
                
                # Group by query type
                for i, query_type in enumerate(query_types):
                    if query_type not in query_results:
                        query_results[query_type] = {
                            'features': [],
                            'pids': [],
                            'camids': []
                        }
                    
                    query_results[query_type]['features'].append(feat[i:i+1].cpu())
                    query_results[query_type]['pids'].append(0)  # Default PID
                    query_results[query_type]['camids'].append(0)  # Default camera ID
        
        # Concatenate features for each query type
        for query_type in query_results:
            if query_results[query_type]['features']:
                query_results[query_type]['features'] = torch.cat(
                    query_results[query_type]['features'], dim=0
                )
                query_results[query_type]['pids'] = np.array(query_results[query_type]['pids'])
                query_results[query_type]['camids'] = np.array(query_results[query_type]['camids'])
        
        return query_results
    
    def _compute_distance_matrix(self, query_features, gallery_features):
        """Compute distance matrix between query and gallery features"""
        # Normalize features
        query_features = torch.nn.functional.normalize(query_features, dim=1)
        gallery_features = torch.nn.functional.normalize(gallery_features, dim=1)
        
        # Compute cosine distance (1 - cosine similarity)
        similarity = torch.mm(query_features, gallery_features.t())
        distmat = 1 - similarity
        
        return distmat
    
    def _evaluate_rank(self, distmat, query_pids, gallery_pids, query_camids, gallery_camids):
        """Evaluate ranking performance"""
        from .metrics import eval_func
        
        # Convert to numpy if needed
        if torch.is_tensor(distmat):
            distmat = distmat.cpu().numpy()
        
        # Evaluate
        cmc, mAP, indices = eval_func(distmat, query_pids, gallery_pids, 
                                     query_camids, gallery_camids)
        
        return cmc, mAP, indices
    
    def generate_ranking_lists(self, results, top_k=100):
        """Generate ranking lists for each query"""
        ranking_results = []
        
        for query_type, result in results.items():
            indices = result['indices']
            
            for i, ranking in enumerate(indices):
                # Get top-k indices
                top_k_indices = ranking[:top_k]
                
                ranking_results.append({
                    'query_idx': i,
                    'query_type': query_type,
                    'ranking_list_idx': top_k_indices.tolist()
                })
        
        return ranking_results
    
    def save_ranking_results(self, results, output_path):
        """Save ranking results to file"""
        ranking_results = self.generate_ranking_lists(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("query_idx\tquery_type\tranking_list_idx\n")
            for result in ranking_results:
                f.write(f"{result['query_idx']}\t{result['query_type']}\t{result['ranking_list_idx']}\n")
        
        self.logger.info(f"Ranking results saved to {output_path}")
        return ranking_results
