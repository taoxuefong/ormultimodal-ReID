import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import torch.nn.functional as F


class MultiModalFusion(nn.Module):
    """Multi-modal fusion module for combining different modalities"""
    
    def __init__(self, embed_dim, fusion_way='concat'):
        super(MultiModalFusion, self).__init__()
        self.embed_dim = embed_dim
        self.fusion_way = fusion_way
        
        if fusion_way == 'concat':
            # Concatenate all modalities and project to original dimension
            self.fusion_proj = nn.Linear(embed_dim * 4, embed_dim)
            nn.init.normal_(self.fusion_proj.weight, std=0.02)
            nn.init.zeros_(self.fusion_proj.bias)
        elif fusion_way == 'attention':
            # Use attention mechanism to fuse modalities
            self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8)
            self.norm = nn.LayerNorm(embed_dim)
        elif fusion_way == 'weighted_sum':
            # Learnable weighted sum of modalities
            self.weights = nn.Parameter(torch.ones(4))  # 4 modalities
            self.softmax = nn.Softmax(dim=0)
    
    def forward(self, features_input):
        """
        Args:
            features_input: dict or list of features for each modality
                - 'text': text features [B, embed_dim]
                - 'nir': NIR features [B, embed_dim]
                - 'cp': CP features [B, embed_dim]
                - 'sk': SK features [B, embed_dim]
        """
        # 处理字典输入
        if isinstance(features_input, dict):
            features_list = list(features_input.values())
        else:
            features_list = features_input
            
        if len(features_list) == 0:
            return torch.zeros(1, self.embed_dim).to(features_list[0].device) if features_list else torch.zeros(1, self.embed_dim)

        # 过滤掉非张量元素
        features_list = [feat for feat in features_list if isinstance(feat, torch.Tensor)]
        if len(features_list) == 0:
            return torch.zeros(1, self.embed_dim).to(features_list[0].device) if features_list else torch.zeros(1, self.embed_dim)

        if self.fusion_way == 'concat':
            # Concatenate all features
            combined = torch.cat(features_list, dim=1)
            fused = self.fusion_proj(combined)
        elif self.fusion_way == 'attention':
            # Stack features and use self-attention
            stacked = torch.stack(features_list, dim=1)  # [B, N, embed_dim]
            attended, _ = self.attention(stacked, stacked, stacked)
            fused = self.norm(attended.mean(dim=1))  # Average pooling
        elif self.fusion_way == 'weighted_sum':
            # Weighted sum of features
            weights = self.softmax(self.weights)
            fused = sum(w * f for w, f in zip(weights, features_list))
        else:
            # Simple addition
            fused = sum(features_list)
            
        return fused


class MultiModalReID(nn.Module):
    """Multi-modal person re-identification model"""
    
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.fusion_way = getattr(args, 'fusion_way', 'concat')
        
        # Build CLIP base model
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice, args.img_size, args.stride_size
        )
        self.embed_dim = base_cfg['embed_dim']
        
        # Temperature parameter for contrastive learning
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / args.temperature))
        
        # Multi-modal fusion module
        self.fusion = MultiModalFusion(self.embed_dim, self.fusion_way)
        
        # ID classifier
        if 'id' in getattr(args, 'loss_names', ''):
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)
    
    def encode_image(self, image):
        """Encode RGB images"""
        return self.base_model.encode_image(image)
    
    def encode_text(self, text):
        """Encode text"""
        return self.base_model.encode_text(text)
    
    def encode_multimodal_query(self, batch):
        """Encode multi-modal query"""
        features_dict = {}
        
        # Encode text if present
        if batch.get('text_tokens') is not None:
            text_feats = self.encode_text(batch['text_tokens'])
            # Get the last token (EOS token) for text representation
            text_feat = text_feats[:, -1, :]  # [B, embed_dim]
            features_dict['text'] = text_feat
        
        # Encode visual modalities if present
        if batch.get('nir_images') is not None:
            nir_feats = self.encode_image(batch['nir_images'])
            features_dict['nir'] = nir_feats[:, 0, :]  # [B, embed_dim]
        
        if batch.get('cp_images') is not None:
            cp_feats = self.encode_image(batch['cp_images'])
            features_dict['cp'] = cp_feats[:, 0, :]  # [B, embed_dim]
        
        if batch.get('sk_images') is not None:
            sk_feats = self.encode_image(batch['sk_images'])
            features_dict['sk'] = sk_feats[:, 0, :]  # [B, embed_dim]
        
        # Fuse modalities
        fused_feat = self.fusion(features_dict)
        
        return fused_feat
    
    def forward(self, batch):
        """Forward pass"""
        ret = {}
        
        if self.training:
            # Training mode
            vis_images = batch['vis_images']
            nir_images = batch['nir_images']
            cp_images = batch['cp_images']
            sk_images = batch['sk_images']
            caption_ids = batch['caption_ids']
            labels = batch['pids']
            
            # Encode all modalities
            vis_feats = self.encode_image(vis_images)[:, 0, :]  # [B, embed_dim]
            nir_feats = self.encode_image(nir_images)[:, 0, :]  # [B, embed_dim]
            cp_feats = self.encode_image(cp_images)[:, 0, :]  # [B, embed_dim]
            sk_feats = self.encode_image(sk_images)[:, 0, :]  # [B, embed_dim]
            text_feats = self.encode_text(caption_ids)[:, -1, :]  # [B, embed_dim]
            
            # Create features dictionary for fusion
            features_dict = {
                'text': text_feats,
                'nir': nir_feats,
                'cp': cp_feats,
                'sk': sk_feats
            }
            
            # Fuse query modalities
            fused_feat = self.fusion(features_dict)
            
            # Compute contrastive loss
            logit_scale = self.logit_scale.exp()
            ret['temperature'] = 1 / logit_scale
            
            # Image-to-fused query contrastive loss
            ret['itc_loss'] = self.compute_contrastive_loss(vis_feats, fused_feat, logit_scale)
            
            # Individual modality losses for better training
            ret['nir_loss'] = self.compute_contrastive_loss(vis_feats, nir_feats, logit_scale)
            ret['cp_loss'] = self.compute_contrastive_loss(vis_feats, cp_feats, logit_scale)
            ret['sk_loss'] = self.compute_contrastive_loss(vis_feats, sk_feats, logit_scale)
            ret['text_loss'] = self.compute_contrastive_loss(vis_feats, text_feats, logit_scale)
            
            # ID classification loss
            if hasattr(self, 'classifier'):
                id_logits = self.classifier(fused_feat)
                ret['id_loss'] = nn.CrossEntropyLoss()(id_logits, labels)
            
        else:
            # Inference mode
            if 'vis_images' in batch:
                # Gallery images
                vis_feats = self.encode_image(batch['vis_images'])[:, 0, :]
                ret['gallery_features'] = vis_feats
            else:
                # Query
                fused_feat = self.encode_multimodal_query(batch)
                ret['query_features'] = fused_feat
                
                # 调试信息：打印特征维度
                # print(f"DEBUG: fused_feat size: {fused_feat.size()}")
                # print(f"DEBUG: self.embed_dim: {self.embed_dim}")
                # print(f"DEBUG: fused_feat.size(-1): {fused_feat.size(-1)}")
                
                # 确保 query 特征维度与 gallery 特征维度一致
                if hasattr(self, 'embed_dim'):
                    if fused_feat.size(-1) != self.embed_dim:
                        # print(f"DEBUG: Dimension mismatch detected! Creating projection layer...")
                        # 如果维度不匹配，进行投影
                        if not hasattr(self, 'query_proj'):
                            self.query_proj = nn.Linear(fused_feat.size(-1), self.embed_dim).to(fused_feat.device)
                            # print(f"DEBUG: Created projection layer: {fused_feat.size(-1)} -> {self.embed_dim}")
                        fused_feat = self.query_proj(fused_feat)
                        ret['query_features'] = fused_feat
                        # print(f"DEBUG: After projection: {fused_feat.size()}")
                    # else:
                        # print(f"DEBUG: Dimensions match, no projection needed")
        
        return ret
    
    def compute_contrastive_loss(self, image_feats, query_feats, logit_scale):
        # 确保所有张量都在同一个设备上
        device = image_feats.device
        query_feats = query_feats.to(device)
        logit_scale = logit_scale.to(device)
        
        # 确保所有张量的 dtype 一致
        dtype = image_feats.dtype
        query_feats = query_feats.to(dtype=dtype)
        logit_scale = logit_scale.to(dtype=dtype)
        
        # 归一化特征
        image_feats = F.normalize(image_feats, dim=-1)
        query_feats = F.normalize(query_feats, dim=-1)
        
        # 计算相似度矩阵
        logits = logit_scale * image_feats @ query_feats.t()
        
        # 确保 labels 的值范围正确
        B_image = image_feats.shape[0]
        B_query = query_feats.shape[0]
        
        # 确保 logits 的形状与 labels_image 匹配
        if logits.shape[1] < B_image:
            # 如果 logits 的第二维小于 B_image，则需要调整 labels_image
            labels_image = torch.arange(logits.shape[1], device=device)
        else:
            labels_image = torch.arange(B_image, device=device)
            if logits.shape[1] > B_image:
                logits = logits[:, :B_image]
        
        loss_i2q = F.cross_entropy(logits, labels_image)
        
        # 确保 logits.t() 和 labels 的批次大小一致且值范围正确
        # 确保 logits.t() 的形状与 labels_query 匹配
        if logits.t().shape[1] < B_query:
            labels_query = torch.arange(logits.t().shape[1], device=device)
        else:
            labels_query = torch.arange(B_query, device=device)
            if logits.t().shape[1] > B_query:
                logits_t = logits.t()[:, :B_query]
            else:
                logits_t = logits.t()
        
        loss_q2i = F.cross_entropy(logits_t, labels_query)
        
        loss = (loss_i2q + loss_q2i) / 2
        return loss


def build_multimodal_model(args, num_classes=11003):
    """Build multi-modal ReID model"""
    model = MultiModalReID(args, num_classes)
    # Convert model to fp16 if needed
    convert_weights(model)
    return model
