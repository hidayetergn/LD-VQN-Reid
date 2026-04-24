import torchvision.transforms as T
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import random
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data.sampler import Sampler

from .veri776_dataset import VeRi776Dataset
from .vehicleid_dataset import VehicleIDDataset

class RandomIdentitySampler(Sampler):
    """
    Randomly samples N identities, then samples K images per identity to 
    form a mini-batch of size N * K. Crucial for the Triplet Loss.
    """
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        
        self.index_dic = defaultdict(list)
        for index, item in enumerate(self.data_source.data):
            pid = int(item['id'])
            self.index_dic[pid].append(index)
            
        self.pids = list(self.index_dic.keys())

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = self.index_dic[pid]
            # If an identity has fewer images than num_instances, sample with replacement
            if len(idxs) < self.num_instances:
                batch_idxs_dict[pid] = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
            else:
                batch_idxs_dict[pid] = np.random.choice(idxs, size=self.num_instances, replace=False).tolist()

        avai_pids = list(self.pids)
        random.shuffle(avai_pids)
        
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = avai_pids[:self.num_pids_per_batch]
            avai_pids = avai_pids[self.num_pids_per_batch:]
            
            for pid in selected_pids:
                final_idxs.extend(batch_idxs_dict[pid])
                
        return iter(final_idxs)

    def __len__(self):
        return (len(self.pids) // self.num_pids_per_batch) * self.batch_size


def build_transforms(cfg, is_train=True):
    """Constructs torchvision transforms based on YAML config."""
    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if is_train:
        transform = T.Compose([
            T.Resize(cfg['DATA']['IMG_RESIZE']),
            T.RandomCrop(cfg['DATA']['IMG_CROP']),
            T.RandomHorizontalFlip(p=0.5 if cfg['DATA']['RANDOM_FLIP'] else 0.0),
            T.ToTensor(),
            normalize_transform
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg['DATA']['IMG_CROP']), # Direct resize for testing to avoid crop variance
            T.ToTensor(),
            normalize_transform
        ])
    return transform

def make_dataloader(cfg):
    """Factory function to build train and test dataloaders."""
    
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    
    if cfg['DATA']['DATASET'] == "VeRi-776":
        train_set = VeRi776Dataset(cfg['DATA']['TRAIN_JSONL'], cfg['DATA']['ROOT_DIR'] + '/image_train', 
                                   train_transforms, tokenizer, cfg['DATA']['MAX_SEQ_LENGTH'])
        query_set = VeRi776Dataset(cfg['DATA']['QUERY_JSONL'], cfg['DATA']['ROOT_DIR'] + '/image_query', 
                                   val_transforms, tokenizer, cfg['DATA']['MAX_SEQ_LENGTH'])
        gallery_set = VeRi776Dataset(cfg['DATA']['TEST_JSONL'], cfg['DATA']['ROOT_DIR'] + '/image_test', 
                                     val_transforms, tokenizer, cfg['DATA']['MAX_SEQ_LENGTH'])
        
        val_sets = {'query': query_set, 'gallery': gallery_set}
        
    elif cfg['DATA']['DATASET'] == "VehicleID":
        train_set = VehicleIDDataset(cfg['DATA']['TRAIN_JSONL'], cfg['DATA']['ROOT_DIR'] + '/image', 
                                     train_transforms, tokenizer, cfg['DATA']['MAX_SEQ_LENGTH'])
        
        # Build a dict of dataloaders for progressive testing (800, 1600, etc.)
        val_sets = {}
        for split_path in cfg['DATA']['TEST_SPLITS']:
            split_name = os.path.basename(split_path).split('.')[0]
            val_sets[split_name] = VehicleIDDataset(split_path, cfg['DATA']['ROOT_DIR'] + '/image', 
                                                    val_transforms, tokenizer, cfg['DATA']['MAX_SEQ_LENGTH'])
    else:
        raise ValueError(f"Unsupported dataset: {cfg['DATA']['DATASET']}")

    # Build Train Loader with P*K Sampler
    train_sampler = RandomIdentitySampler(train_set, cfg['DATALOADER']['BATCH_SIZE'], cfg['DATALOADER']['NUM_INSTANCES'])
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg['DATALOADER']['BATCH_SIZE'], 
        sampler=train_sampler, 
        num_workers=cfg['DATALOADER']['NUM_WORKERS'],
        pin_memory=True,
        drop_last=True
    )

    # Build Validation Loader(s)
    val_loaders = {}
    for name, vset in val_sets.items():
        val_loaders[name] = DataLoader(
            vset, 
            batch_size=cfg['DATALOADER']['BATCH_SIZE'], 
            shuffle=False, 
            num_workers=cfg['DATALOADER']['NUM_WORKERS'],
            pin_memory=True
        )

    return train_loader, val_loaders