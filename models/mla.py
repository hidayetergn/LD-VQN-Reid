import torch
import torch.nn as nn

class MultiLevelAdapter(nn.Module):
    """
    Multi-Level Adapter (MLA) for hierarchical visual feature extraction.
    Recovers early-stage textural primitives and mid-level geometric structures
    by fusing hidden states from multiple transformer depths.
    
    Reference: Section 3.2.1, Equations 2-3.
    """
    def __init__(self, embed_dim=768, bottleneck_factor=4, num_anchor_layers=5):
        super().__init__()
        self.embed_dim = embed_dim
        self.reduced_dim = embed_dim // bottleneck_factor
        self.num_layers = num_anchor_layers
        
        # Bank of learnable linear adapters for each hierarchical depth (Equation 2)
        self.adapters = nn.ModuleList([
            nn.Sequential(
    



    def forward(self, hidden_states: list) -> torch.Tensor:
        """
        Args:
            hidden_states (list): List of tensors from anchor layers, 
                                  each of shape (B, N_v, D).
        Returns:
            f_img (torch.Tensor): Integrated multi-scale visual features, shape (B, N_v, D).
        """
        assert len(hidden_states) == self.num_layers, \
            f"Expected {self.num_layers} layers, got {len(hidden_states)}"
        
        projected_states = []
        for i, state in enumerate(hidden_states):
    


