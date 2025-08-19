import logging
import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from datasets.balanced_sampler import PRCVBalancedSampler
from torch.utils.data.distributed import DistributedSampler

from utils.comm import get_world_size

from .bases import ImageDataset, SketchDataset, ImageTextMSMDataset, ImageTextMSMMLMDataset, TextDataset, ImageTextDataset, ImageTextMCQDataset, ImageTextMaskColorDataset, ImageTextMLMDataset, ImageTextMCQMLMDataset, SketchTextDataset, MultiModalDataset, MultiModalQueryDataset, PRCVTrainDataset

from .f30k import F30K
from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid
from .prcv import PRCV
__factory = {'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES, 'F30K': F30K, 'RSTPReid': RSTPReid, 'PRCV': PRCV}


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # transform for training
    if aug:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform


def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
             batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict

def multimodal_collate(batch):
    """专门用于多模态数据的collate函数"""
    keys = set([key for b in batch for key in b.keys()])
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
            # 只处理有数据的张量，跳过None值
            valid_tensors = [t for t in v if t is not None]
            if valid_tensors:
                # 检查所有有效张量是否形状一致
                shapes = [t.shape for t in valid_tensors]
                if len(set(shapes)) == 1:
                    # 形状一致，可以stack
                    batch_tensor_dict.update({k: torch.stack(valid_tensors)})
                else:
                    # 形状不一致，跳过这个键
                    print(f"WARNING: Skipping key '{k}' due to inconsistent tensor shapes: {shapes}")
                    continue
            else:
                # 所有张量都是None，跳过这个键
                continue
        elif isinstance(v[0], str):
            batch_tensor_dict.update({k: v})
        elif isinstance(v[0], list):
            batch_tensor_dict.update({k: v})
        else:
            batch_tensor_dict.update({k: v})

    return batch_tensor_dict

def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("CLIP2ReID.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir, nlp_aug=args.nlp_aug)

    if args.training:
        train_transforms = build_transforms(img_size=args.img_size,
                                            aug=args.img_aug,
                                            is_train=True)
        val_transforms = build_transforms(img_size=args.img_size,
                                          is_train=False)

        if args.MCQ:
            train_set = ImageTextMCQDataset(dataset.train,
                                     train_transforms,
                                     text_length=args.text_length)
        elif args.MCM:
            train_set = ImageTextMaskColorDataset(dataset.train,
                                     train_transforms,
                                     text_length=args.text_length,
                                     masked_token_rate=args.masked_token_rate,
                                     masked_token_unchanged_rate=args.masked_token_unchanged_rate)
        elif args.MLM:
            train_set = ImageTextMLMDataset(dataset.train,
                                     train_transforms,
                                     text_length=args.text_length)
        elif args.MSM:
            train_set = ImageTextMSMDataset(dataset.train,
                                     train_transforms,
                                     text_length=args.text_length)
        elif args.MCQMLM:
            train_set = ImageTextMCQMLMDataset(dataset.train,
                                     train_transforms,
                                     text_length=args.text_length)
        elif args.MSMMLM:
            train_set = ImageTextMSMMLMDataset(dataset.train,
                                     train_transforms,
                                     text_length=args.text_length)
        else:
            if args.dataset_name == 'PRCV':
                train_set = PRCVTrainDataset(dataset.train,
                                         train_transforms,
                                         text_length=args.text_length)
            else:
                train_set = ImageTextDataset(dataset.train,
                                         train_transforms,
                                         text_length=args.text_length)

        num_classes = len(dataset.train_id_container)

        if args.sampler == 'identity':
            if args.distributed:
                logger.info('using ddp random identity sampler')
                logger.info('DISTRIBUTED TRAIN START')
                mini_batch_size = args.batch_size // get_world_size()
                # TODO wait to fix bugs
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, args.batch_size, args.num_instance)
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True)
                # sampler = DistributedSampler(train_set)
                # train_loader = DataLoader(
                #     train_set,
                #     num_workers=num_workers,
                #     # sampler=sampler,
                #     # batch_size=mini_batch_size,
                #     batch_sampler=batch_sampler,
                #     collate_fn=collate)
            else:
                logger.info(
                    f'using random identity sampler: batch_size: {args.batch_size}, id: {args.batch_size // args.num_instance}, instance: {args.num_instance}'
                )
                train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdentitySampler(
                                              dataset.train, args.batch_size,
                                              args.num_instance),
                                          num_workers=num_workers,
                                          collate_fn=collate)
        elif args.sampler == 'random':
            # TODO add distributed condition
            logger.info('using random sampler')
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      collate_fn=collate)
        elif args.sampler == 'balanced':
            logger.info('using balanced sampler')
            # 计算每个batch的类别数和每个类别的样本数
            n_classes_per_batch = args.batch_size // args.num_instance
            n_samples_per_class = args.num_instance
            
            logger.info(f'Balanced sampling: {n_classes_per_batch} classes per batch, {n_samples_per_class} samples per class')
            
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      sampler=PRCVBalancedSampler(
                                          dataset.train, n_classes_per_batch, n_samples_per_class),
                                      num_workers=num_workers,
                                      collate_fn=collate)
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(args.sampler))

        # use test set as validate set
        ds = dataset.val if args.val_dataset == 'val' else dataset.test

        if args.dataset_name == 'PRCV':
            val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'], ds['image_ids'],
                                       val_transforms)
            val_query_set = MultiModalQueryDataset(ds['queries'], val_transforms,
                                          text_length=args.text_length,
                                          base_dir=os.path.join(args.root_dir, 'val'))
            val_sketch_set = None  # Not needed for PRCV
        else:
            val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'], ds['image_ids'],
                                       val_transforms)
            val_txt_set = SketchTextDataset(ds['simg_paths'], ds['simage_ids'], ds['caption_pids'],
                                      ds['captions'], val_transforms,
                                      text_length=args.text_length)
            val_sketch_set = SketchDataset(ds['simg_paths'], ds['simage_ids'], ds['simage_pids'], val_transforms)
                                

        val_img_loader = DataLoader(val_img_set,
                                    batch_size=args.test_batch_size if args.dataset_name == 'PRCV' else args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        
        if args.dataset_name == 'PRCV':
            val_txt_loader = DataLoader(val_query_set,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        collate_fn=multimodal_collate)
            val_sketch_loader = None
        else:
            val_txt_loader = DataLoader(val_txt_set,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=num_workers)
            val_sketch_loader = DataLoader(val_sketch_set,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=num_workers)

        return train_loader, val_img_loader, val_txt_loader, val_sketch_loader, num_classes

    else:
        # build dataloader for testing
        if tranforms:
            test_transforms = tranforms
        else:
            test_transforms = build_transforms(img_size=args.img_size,
                                               is_train=False)

        ds = dataset.test
        test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'], ds['image_ids'],
                                    test_transforms)
        
        if args.dataset_name == 'PRCV':
            test_txt_set = MultiModalQueryDataset(ds['queries'], test_transforms,
                                       text_length=args.text_length,
                                       base_dir=os.path.join(args.root_dir, 'val'))
            test_sketch_loader = DataLoader(test_txt_set,
                                        batch_size=args.test_batch_size,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        collate_fn=multimodal_collate)
            test_sketch_set = None
        else:
            test_txt_set = SketchTextDataset(ds['simg_paths'], ds['simage_ids'], ds['caption_pids'],
                                       ds['captions'], test_transforms, 
                                       text_length=args.text_length)
            test_sketch_set = SketchDataset(ds['simg_paths'], ds['simage_ids'], ds['simage_pids'], test_transforms)

        test_img_loader = DataLoader(test_img_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     collate_fn=multimodal_collate)
        if args.dataset_name == 'PRCV':
            test_sketch_loader = None
        else:
            test_sketch_loader = DataLoader(test_sketch_set,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=num_workers)

        return test_img_loader, test_txt_loader, test_sketch_loader