import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class DynamicRoutingAdjacency(nn.Module):
    """
    Generates a dynamic topological adjacency matrix directly from the unconstrained
    text prompt. (Reference: Equation 4)
    """
    def __init__(self, embed_dim, num_components, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_components = num_components
        



……………….

    def forward(self, text_global: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_global: Mean pooled text representation (B, D)
        Returns:
            A: Adjacency matrix (B * H, M, M) normalized via Sigmoid
        """
        B = text_global.shape[0]
        # Generate raw weights
        raw_adj = self.mlp(text_global) # (B, H * M * M)
     

class DTCRBlock(nn.Module):
    """
    Dynamic Text-Conditioned Routing Module.
    Establishes text-guided latent semantic nodes and selectively routes linguistic
    primitives to spatial coordinates. (Reference: Section 3.2.2)
    """
    def __init__(self, embed_dim=768, num_components=16, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_components = num_components
        self.num_heads = num_heads
        
        # Learnable Component Nodes C \in \mathbb{R}^{M \times D}
        self.components = nn.Parameter(torch.empty(1, num_components, embed_dim))
        nn.init.xavier_uniform_(self.components)
        
        self.adjacency_generator = DynamicRoutingAdjacency(embed_dim, num_components, num_heads)
        
        # Stage 1: Intra-Component Message Passing (GSA)
        self.gsa = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Stage 2: Visual Grounding (Cross-Image Attention)
        self.cia = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Stage 3: Textual Refinement (Cross-Text Attention)
        self.cta = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)





    def forward(self, T: torch.Tensor, V: torch.Tensor, text_mask: torch.Tensor = None):
        """
        Args:
            T: Text embeddings (B, N_t, D)
            V: Multi-scale visual embeddings from MLA (B, N_v, D)
            text_mask: Padding mask for text tokens (B, N_t)
        Returns:
            T_out: Refined text embeddings (B, N_t, D)
            C_2: Visually grounded component nodes for Orthogonal Loss (B, M, D)
        """
        B = T.shape[0]
        
        # Expand components to batch size
        C = self.components.expand(B, -1, -1) # (B, M, D)
        
        # 1. Generate dynamic adjacency matrix based on global text context
        # Convert text padding mask to binary float mask for pooling
        mask_float = text_mask.unsqueeze(-1).float()
        t_global = (T * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)
        
        # A matrix shape: (B * H, M, M)
        A = self.adjacency_generator(t_global) 
        
        # Transform A into an attention mask suitable for PyTorch's MHA
        # PyTorch MHA expects additive mask (0 for keep, -inf for ignore). 
        # Here we use A as an explicit attention weight multiplier, which requires custom
        # handling or modifying the standard MHA. For TPAMI elegance, we approximate
        # by adjusting the attention scores internally or using a customized MHA.
        # *Note for standard PyTorch API*: We convert A (0 to 1) to an additive mask.
        attn_mask = (1.0 - A) * -1e4 
        
        # Equation 5: C^{(1)} = Norm(C + GSA(C, C, C, mask=A))
        # Note: We repeat attn_mask to match batch sizes required by MHA if needed
        
        
