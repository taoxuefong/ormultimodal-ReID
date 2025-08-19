import os
import os.path as op
from typing import List
import json

from utils.iotools import read_json
from .bases import BaseDataset
from prettytable import PrettyTable


class PRCV(BaseDataset):
    """
    PRCV Dataset for multi-modal person re-identification
    
    Supports four modalities:
    - NIR (Near-Infrared)
    - CP (Colored Pencil)
    - SK (Sketch)
    - TEXT (Text description)
    
    Target modality: RGB (VIS)
    """
    dataset_dir = 'PRCV'

    def __init__(self, root='', nlp_aug=False, verbose=True):
        super(PRCV, self).__init__()
        # 直接使用root作为数据集根目录，避免重复的PRCV
        self.dataset_dir = root
        self.train_dir = op.join(self.dataset_dir, 'train')
        self.val_dir = op.join(self.dataset_dir, 'val')
        
        # Modality directories - 这些目录在训练时不需要，因为file_path已经包含了完整路径
        # self.nir_dir = op.join(self.train_dir, 'nir')
        # self.cp_dir = op.join(self.train_dir, 'cp')
        # self.sk_dir = op.join(self.train_dir, 'sk')
        # self.vis_dir = op.join(self.train_dir, 'vis')
        self.text_anno_path = op.join(self.train_dir, 'text_annos.json')
        
        # Validation directories
        self.val_gallery_dir = op.join(self.val_dir, 'gallery')
        self.val_nir_dir = op.join(self.val_dir, 'nir')
        self.val_cp_dir = op.join(self.val_dir, 'cp')
        self.val_sk_dir = op.join(self.val_dir, 'sk')
        self.val_queries_path = op.join(self.val_dir, 'val_queries.json')
        
        self._check_before_run()

        self.train_annos, self.val_queries = self._load_data()

        self.train, self.train_id_container = self._process_train_anno(self.train_annos, training=True)
        self.val, self.val_id_container = self._process_val_queries(self.val_queries)
        
        # 为了兼容性，将val也设置为test
        self.test = self.val
        self.test_id_container = self.val_id_container

        if verbose:
            self.logger.info("=> PRCV Dataset loaded")
            self.show_dataset_info()

    def _check_before_run(self):
        """Check if all required directories and files exist"""
        if not op.exists(self.train_dir):
            raise RuntimeError(f"Train directory not found: {self.train_dir}")
        if not op.exists(self.val_dir):
            raise RuntimeError(f"Validation directory not found: {self.val_dir}")
        if not op.exists(self.text_anno_path):
            raise RuntimeError(f"Text annotations not found: {self.text_anno_path}")
        if not op.exists(self.val_queries_path):
            raise RuntimeError(f"Validation queries not found: {self.val_queries_path}")

    def _load_data(self):
        """Load training annotations and validation queries"""
        # Load training text annotations
        with open(self.text_anno_path, 'r', encoding='utf-8') as f:
            train_annos = json.load(f)
        
        # Load validation queries
        with open(self.val_queries_path, 'r', encoding='utf-8') as f:
            val_queries = json.load(f)
            
        return train_annos, val_queries

    def _process_train_anno(self, annos: List[dict], training=True):
        """Process training annotations"""
        pid_container = set()
        dataset = []
        image_id = 0
        
        for anno in annos:
            pid = int(anno['id'])
            pid_container.add(pid)
            
            # RGB image path - file_path已经包含了vis/子目录
            # 例如：file_path = "vis/0001/0001_llcm_0001_c03_s184251_f31995_vis.jpg"
            vis_path = op.join(self.train_dir, anno['file_path'])
            caption = anno['caption']
            
            # 从vis路径构建其他模态的路径
            # 例如：vis/0001/0001_llcm_0001_c03_s184251_f31995_vis.jpg
            # 转换为：nir/0001/0001_llcm_0001_c01_s185535_f7000_nir.jpg
            # 注意：文件名不是一一对应的，所以我们需要从同一类别中随机选择
            
            # 提取类别目录和基础文件名
            path_parts = anno['file_path'].split('/')
            class_dir = path_parts[1]  # 0001
            base_name = path_parts[2].split('_')  # ['0001', 'llcm', '0001', 'c03', 's184251', 'f31995', 'vis.jpg']
            
            # 构建其他模态的目录路径
            nir_dir = op.join(self.train_dir, 'nir', class_dir)
            cp_dir = op.join(self.train_dir, 'cp', class_dir)
            sk_dir = op.join(self.train_dir, 'sk', class_dir)
            
            # 尝试找到对应模态的图像文件
            nir_path = None
            cp_path = None
            sk_path = None
            
            # 查找NIR图像（查找包含'nir'的文件）
            if os.path.exists(nir_dir):
                nir_files = [f for f in os.listdir(nir_dir) if 'nir' in f.lower()]
                if nir_files:
                    nir_path = op.join(nir_dir, nir_files[0])  # 使用第一个找到的NIR文件
            
            # 查找CP图像（查找包含'colorpencil'或'cp'的文件）
            if os.path.exists(cp_dir):
                cp_files = [f for f in os.listdir(cp_dir) if any(x in f.lower() for x in ['colorpencil', 'cp'])]
                if cp_files:
                    cp_path = op.join(cp_dir, cp_files[0])  # 使用第一个找到的CP文件
            
            # 查找SK图像（查找包含'sketch'或'sk'的文件）
            if os.path.exists(sk_dir):
                sk_files = [f for f in os.listdir(sk_dir) if any(x in f.lower() for x in ['sketch', 'sk'])]
                if sk_files:
                    sk_path = op.join(sk_dir, sk_files[0])  # 使用第一个找到的SK文件
            
            # 如果找不到对应模态的图像，使用RGB图像作为后备
            if nir_path is None:
                nir_path = vis_path
            if cp_path is None:
                cp_path = vis_path
            if sk_path is None:
                sk_path = vis_path
            
            dataset.append((pid, image_id, vis_path, nir_path, cp_path, sk_path, caption))
            image_id += 1
        
        # Sort by pid to ensure consistency
        pid_container = sorted(pid_container)
        return dataset, pid_container

    def _process_val_queries(self, queries: List[dict]):
        """Process validation queries"""
        dataset = {}
        img_paths = []
        image_pids = []
        image_ids = []
        
        # Load gallery images
        gallery_files = sorted([f for f in os.listdir(self.val_gallery_dir) if f.endswith('.jpg')])
        for i, filename in enumerate(gallery_files):
            img_path = op.join(self.val_gallery_dir, filename)
            img_paths.append(img_path)
            image_pids.append(i + 1)  # Assuming sequential IDs starting from 1
            image_ids.append(i)
        
        dataset['img_paths'] = img_paths
        dataset['image_pids'] = image_pids
        dataset['image_ids'] = image_ids
        
        # Process queries
        query_data = []
        for query in queries:
            query_idx = query['query_idx']
            query_type = query['query_type']
            content = query['content']
            
            # Parse query type to determine modalities
            modalities = self._parse_query_type(query_type)
            
            # Create query entry
            query_entry = {
                'query_idx': query_idx,
                'query_type': query_type,
                'modalities': modalities,
                'content': content
            }
            query_data.append(query_entry)
        
        dataset['queries'] = query_data
        dataset['query_types'] = list(set([q['query_type'] for q in query_data]))
        
        return dataset, set(image_pids)

    def _parse_query_type(self, query_type: str):
        """Parse query type to determine which modalities are present"""
        # 按照query_type字符串中模态出现的实际顺序来添加模态
        # 这样可以确保modalities的顺序与content的顺序一致
        
        # 从query_type字符串中按顺序提取模态
        # 例如: "threemodal_TEXT_NIR_CP" -> ['text', 'nir', 'cp']
        # 或者 "twomodal_NIR_TEXT" -> ['nir', 'text']
        
        modalities = []
        # 将query_type按下划线分割，找到模态部分
        parts = query_type.split('_')
        # 模态通常在后半部分，找到所有大写字母的单词
        for part in parts:
            if part in ['TEXT', 'NIR', 'CP', 'SK']:
                if part == 'TEXT':
                    modalities.append('text')
                elif part == 'NIR':
                    modalities.append('nir')
                elif part == 'CP':
                    modalities.append('cp')
                elif part == 'SK':
                    modalities.append('sk')
        
        # print(f"DEBUG: query_type: {query_type}")
        # print(f"DEBUG: parsed modalities: {modalities}")
        
        return modalities

    def show_dataset_info(self):
        """Display dataset statistics"""
        num_train_pids = len(self.train_id_container)
        num_train_samples = len(self.train)
        num_val_pids = len(self.val_id_container)
        num_val_queries = len(self.val['queries'])
        num_gallery = len(self.val['img_paths'])
        
        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'samples'])
        table.add_row(['train', num_train_pids, num_train_samples])
        table.add_row(['val_gallery', num_val_pids, num_gallery])
        table.add_row(['val_queries', '-', num_val_queries])
        self.logger.info('\n' + str(table))
        
        # Show query type distribution
        query_types = self.val['query_types']
        self.logger.info(f"Query types: {query_types}")
        
        # 显示测试集信息（与验证集相同）
        self.logger.info(f"Test set: same as validation set")
