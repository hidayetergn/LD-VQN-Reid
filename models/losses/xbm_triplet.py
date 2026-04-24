import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossBatchMemory(nn.Module):
    """
    Cross-Batch Memory (XBM) mechanism to maintain a dynamic FIFO queue of recent 
    embeddings, effectively expanding the receptive field of the Batch Hard Triplet 
    Loss without computational overhead.
    
    Reference: Section 3.2, Algorithm 1.
    """
    def __init__(self, memory_size=5000, feature_dim=768):
        super().__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        




    @torch.no_grad()
    def enqueue_dequeue(self, batch_features: torch.Tensor, batch_labels: torch.Tensor):
        """Updates the memory queue in a First-In-First-Out (FIFO) manner."""
        batch_size = batch_features.shape[0]
        ptr = int(self.ptr.item())

        # If the batch exceeds the remaining memory space, wrap around
        if ptr + batch_size > self.memory_size:
            remaining = self.memory_size - ptr
            self.memory_features[ptr:] = batch_features[:remaining].detach()
            self.memory_labels[ptr:] = batch_labels[:remaining].detach()
            







class XBMBatchHardTripletLoss(nn.Module):
    """
    Computes the Triplet Loss using hardest positive mining within the current batch 
    and hardest negative mining across both the current batch and the XBM memory.
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, features: torch.Tensor, labels: torch.Tensor, 
                memory_features: torch.Tensor = None, memory_labels: torch.Tensor = None):
        """
        Args:
            features: Current batch embeddings (B, D)
            labels: Current batch labels (B,)
            memory_features: Historical embeddings from XBM (M, D)
            memory_labels: Historical labels from XBM (M,)
        """
        # If memory is available, concatenate it with the current batch for negative mining
        if memory_features is not None and memory_labels is not None:
            all_features = torch.cat([features, memory_features], dim=0)
            all_labels = torch.cat([labels, memory_labels], dim=0)
        else:
            all_features = features
            all_labels = labels

        # Compute pairwise distance matrix between current batch and ALL features
        # dist(a, b) = a^2 + b^2 - 2ab
        dot_product = torch.matmul(features, all_features.t())
        square_norm_features = features.pow(2).sum(dim=1, keepdim=True)
        square_norm_all = all_features.pow(2).sum(dim=1, keepdim=True)
        
        dist_mat = square_norm_features + square_norm_all.t() - 2.0 * dot_product
        dist_mat = torch.clamp(dist_mat, min=1e-16).sqrt() # Prevent NaN gradients at 0



