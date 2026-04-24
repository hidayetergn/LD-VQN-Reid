"""
t-SNE Feature Space Visualization Tool for LD-VQN
Reproduces the hard-negative clustering results in Figure 11.
Demonstrates the efficacy of the Orthogonal Disentanglement Loss in preventing feature collapse.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import yaml

from models.ld_vqn import LDVQN
from datasets.build_dataloader import make_dataloader

def get_args():
    parser = argparse.ArgumentParser(description="t-SNE Visualization for LD-VQN Latent Space")
    parser.add_argument('--config', type=str, default='configs/veri776_ldvqn.yml', help="Config file")
    parser.add_argument('--weight', type=str, required=True, help="Path to pre-trained weights")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@torch.no_grad()
def extract_features(model, dataloader, device, num_identities=5):
    """
    Extracts global features for a specific subset of identities.
    """
    features_list = []
    labels_list = []
    
    # We collect a subset to keep the t-SNE plot clean and interpretable
    collected_pids = set()

    for batch in dataloader:
        images = batch['image'].to(device)
        pids = batch['pid'].numpy()
        
        # Only process if we need more identities or the identity is already being tracked
        valid_indices = []
        for i, pid in enumerate(pids):

            
        images = images[valid_indices]
        pids = pids[valid_indices]

        # For t-SNE, we evaluate the visual branch purely, or provide dummy text to 
        # extract the visual bottleneck features depending on evaluation protocol
        dummy_input_ids = torch.zeros((images.shape[0], 128), dtype=torch.long).to(device)

        f_bn = model(images, dummy_input_ids, dummy_mask)
        
        features_list.append(f_bn.cpu().numpy())
        labels_list.extend(pids)
        
        # Break early if we have enough samples (e.g., 20 samples per ID)
        if len(labels_list) > num_identities * 20:
            break

    return np.vstack(features_list), np.array(labels_list)

def plot_tsne(features, labels, output_path):
    """
    Applies t-SNE reduction and plots the distribution using Seaborn.
    Optimized for high-resolution academic publishing.
    """
    print("[*] Computing t-SNE reduction. This may take a moment...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init='pca')
    reduced_features = tsne.fit_transform(features)

    # Styling for TPAMI publication
    plt.figure(figsize=(10, 8), dpi=300)
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "black", "xtick.bottom": True, "ytick.left": True})
    
    # Unique palette for distinct clusters
    palette = sns.color_palette("husl", len(np.unique(labels)))

    sns.scatterplot(
        x=reduced_features[:, 0], 
        y=reduced_features[:, 1],
        hue=labels,
        palette=palette,
        s=120,          # Marker size
        alpha=0.8,      # Transparency
        edgecolor='k',  # Black border around markers for clarity
        linewidth=0.5
    )

    plt.title("Latent Space Disentanglement of Isomorphic Vehicles", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
  

def main():
    args = get_args()
    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LDVQN(cfg).to(device)
    model.load_param(args.weight)
    model.eval()

    # Load gallery set for feature extraction
    _, val_loaders = make_dataloader(cfg)
   
    features, labels = extract_features(model, gallery_loader, device, num_identities=6)
    
    output_path = os.path.join(args.outpu

if __name__ == '__main__':
    main()