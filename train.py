"""
Main Training Script for LD-VQN (Algorithm 1)
Handles dual learning rates, gradient accumulation, Cross-Batch Memory (XBM),
and the synergistic computation of CE, Triplet, and Orthogonal losses.
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from datasets.build_dataloader import make_dataloader
from models.ld_vqn import LDVQN
from models.losses.orthogonal_loss import OrthogonalDisentanglementLoss
from models.losses.xbm_triplet import CrossBatchMemory, XBMBatchHardTripletLoss

def get_args():
    parser = argparse.ArgumentParser(description="Train LD-VQN")
    parser.add_argument('--config', type=str, default='configs/veri776_ldvqn.yml', help='Path to config file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_optimizer(cfg, model):
    """
    Implements the Dual Learning Rate strategy.
    Pre-trained backbones get a smaller LR, custom modules get a larger LR.
    """
    base_lr = float(cfg['SOLVER']['LR_BACKBONE'])


    backbone_params = []
    custom_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "vision_encoder" in name or "text_encoder" in name:
            backbone_params.append(param)
        else:
            # MLA, DTCR, bottleneck, classifier


    optimizer = AdamW([
        {'params': backbone_params, 'lr': base_lr},


    return optimizer

def main():
    args = get_args()
    cfg = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing LD-VQN Training on {device}...")

    # 1. Data Loaders
    train_loader, val_loaders = make_dataloader(cfg)
    
    # Update config with dynamic number of classes based on training set
    num_classes = len(train_loader.dataset.data) if cfg['DATA']['DATASET'] == 'VehicleID' else 776
    cfg['DATA']['NUM_CLASSES'] = num_classes

    # 2. Model Initialization
    model = LDVQN(cfg).to(device)

    # 3. Optimizer & Scheduler
    optimizer = build_optimizer(cfg, model)


    # 4. Loss Functions & Memory
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=cfg['LOSS']['CE_LOSS']['LABEL_SMOOTHING'])

    
    xbm_memory = CrossBatchMemory(
        memory_size=cfg['LOSS']['TRIPLET_LOSS']['XBM_CAPACITY'], 
        feature_dim=768
    ).to(device)
    
    xbm_start_epoch = cfg['LOSS']['TRIPLET_LOSS']['XBM_START_EPOCH']
    lambda_ortho = cfg['LOSS']['ORTHOGONAL_LOSS']['WEIGHT']

    # 5. Training Loop
    best_mAP = 0.0
    os.makedirs("weights", exist_ok=True)

    for epoch in range(1, cfg['SOLVER']['MAX_EPOCHS'] + 1):
        model.train()

        
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['SOLVER']['MAX_EPOCHS']}")
        
        for step, batch in enumerate(pbar):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)


            # Forward pass
            outputs = model(images, input_ids, attention_mask)
            
            global_feat = outputs['global_feat']
            logits = outputs['logits']


            # Compute Individual Losses
            loss_ce = criterion_ce(logits, labels)
 

            # XBM Augmented Triplet Loss Logic (Algorithm 1)
            if epoch > xbm_start_epoch:
                mem_feat, mem_labels = xbm_memory.get_memory()
                loss_tri = criterion_triplet(global_feat, labels, mem_feat, mem_labels)
     

            # Combine and scale by accumulation steps
            loss = (loss_ce + loss_tri +
            
            # Backward pass
            loss.backward()

            # Gradient Accumulation & Clipping
            if (step + 1) % accumulate_steps == 0 or (step + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)


            # Enqueue current batch features to XBM
            xbm_memory.enqueue_dequeue(global_feat, labels)

            # Logging metrics
            total_ce += loss_ce.item()


            pbar.set_postfix({
                'CE': f"{total_ce/(step+1):.3f}",


        scheduler.step()

        # 6. Periodic Evaluation (Placeholder structure)
        if epoch % cfg['EVALUATION']['EVAL_PERIOD'] == 0:
            print(f"[*] Triggering Evaluation at Epoch {epoch}...")
            # Here you would instantiate the evaluator class, run k-reciprocal re-ranking,
            # and compute mAP/Rank-1. 
            
            # Simulated dummy metric check for demonstration
            current_mAP = 80.00 # Placeholder for actual evaluated mAP
            
            if current_mAP > best_mAP:
                best_mAP = current_mAP
                save_path = f"weights/ldvqn_{cfg['DATA']['DATASET'].lower()}_best.pth"


if __name__ == '__main__':
    main()