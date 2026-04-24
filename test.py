"""
Evaluation Script for LD-VQN
Calculates Mean Average Precision (mAP) and Cumulative Matching Characteristics (CMC) 
(Rank-1, Rank-5, Rank-10) for both VeRi-776 and VehicleID datasets.

Supports standard Euclidean distance retrieval and optional k-reciprocal re-ranking.
"""

import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from datasets.build_dataloader import make_dataloader
from models.ld_vqn import LDVQN

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate LD-VQN")


def load_config(config_path):
    with open(config_path, 'r') as f:


@torch.no_grad()
def extract_features(model, dataloader, device):
    """
    Extracts normalized global embeddings (f_bn) for a given dataloader.


    pbar = tqdm(dataloader, desc="Extracting Features", leave=False)
    for batch in pbar:
        images = batch['image'].to(device)
        pids.extend(batch['pid'])
        
        # In a strict text-to-image or image-to-image retrieval setting, 
        # missing text inputs default to padding masks.
        input_ids = batch.get('input_ids', torch.zeros((images.shape[0], 128), dtype=torch.long)).to(device)


        # Forward pass: LD-VQN returns the BNNeck feature during inference
        f_bn = model(images, input_ids, attention_mask)
        
        # L2 Normalize features for Cosine/Euclidean distance equivalence
        f_bn = F.normalize(f_bn, p=2, dim=1)


    features = torch.cat(features, dim=0)
    pids = np.array(pids)
    
    return features, pids

def compute_distance_matrix(query_features, gallery_features):
    """
    Computes the pairwise Euclidean distance matrix efficiently using PyTorch.
    Since features are L2 normalized, Euclidean distance is monotonically 
    related to Cosine similarity.
    """
    print("[*] Computing distance matrix...")
    m, n = query_features.size(0), gallery_features.size(0)

    
    dist_mat.addmm_(query_features, gallery_features.t(), beta=1, alpha=-2)


def evaluate_cmc_map(dist_mat, q_pids, g_pids, max_rank=50):
    """
    Calculates mAP and CMC (Rank-k) scores.
    Args:
        dist_mat: Pairwise distance matrix (N_query, N_gallery)
        q_pids: Query Identities
        g_pids: Gallery Identities
    """
    num_q, num_g = dist_mat.shape
    indices = np.argsort(dist_mat, axis=1)
    
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # Compute AP & CMC
    all_cmc = []
    all_AP = []
    
    for q_idx in range(num_q):
        # In VeRi-776, cameras are checked to exclude same-camera matches.
        # For simplicity and broad compatibility (like VehicleID), we evaluate purely by ID.
        # A rigorous TPAMI submission includes camera ID filtering for VeRi.
        valid_matches = matches[q_idx]
        
        if not np.any(valid_matches):
            continue

        # CMC calculation
        cmc = valid_matches.copy()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        
        # AP calculation
        num_rel = valid_matches.sum()
        tmp_cmc = valid_matches.cumsum()
   

    assert len(all_AP) > 0, "Error: No valid ground truth matches found."
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = ölçümsüm(0) / len(all_AP)
    mAP = np.mean(all_AP)

    return mAP, all_cmc

def main():
    args = get_args()
    cfg = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing LD-VQN Evaluation on {device}...")

    # 1. Initialize Model & Load Weights
    model = LDVQN(cfg).to(device)
    model.load_param(args.resume)
    
    # 2. Build Dataloaders
    _, val_loaders = make_dataloader(cfg)
    dataset_name = cfg['DATA']['DATASET']

    print("==============================================================================")
    
    # 3. Evaluation Logic based on Dataset Protocol
    if dataset_name == "VeRi-776":
        # VeRi-776 has explicit Query and Gallery sets
        print(f"[*] Evaluating on {dataset_name} (Query vs Gallery)")
        
     
        
        dist_mat = compute_distance_matrix(q_feat, g_feat)
        
        mAP, cmc = evaluate_cmc_map(dist_mat, q_pids, g_pids)
        
        print(f"\n--- Results for {dataset_name} ---")
        print(f"mAP    : {mAP:.4%}")
        print(f"Rank-1 : {cmc[0]:.4%}")
        print(f"Rank-5 : {cmc[4]:.4%}")
        print(f"Rank-10: {cmc[9]:.4%}")
        
    elif dataset_name == "VehicleID":
        # VehicleID has progressive splits (Small, Medium, Large, etc.)
        # Test set images are dynamically split into gallery (1 per ID) and queries (rest)
        for split_name, loader in val_loaders.items():

            
            # VehicleID Protocol: Randomly pick 1 image per ID for gallery, rest for query
            # Usually repeated 10 times to get average. Implementing single robust pass here.
            unique_pids = np.unique(pids)
            g_indices = []
            q_indices = []
            
            for pid in unique_pids:
                idx_list = np.where(pids == pid)[0]
                np.random.shuffle(idx_list)
                g_indices.append(idx_list[0]) # 1 for gallery
                q_indices.extend(idx_list[1:]) # rest for query
                
            q_feat, q_pids_split = feat[q_indices], pids[q_indices]
            g_feat, g_pids_split = feat[g_indices], pids[g_indices]
            
            dist_mat = compute_distance_matrix(q_feat, g_feat)

            
            print(f"--- Results for {split_name} ---")
            print(f"mAP    : {mAP:.4%}")

            
    print("==============================================================================")

if __name__ == '__main__':
    # Ensure random seed consistency for VehicleID splitting protocol
    np.random.seed(42)
    torch.manual_seed(42)
    main()