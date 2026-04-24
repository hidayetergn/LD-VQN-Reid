import torch
import torch.nn as nn
from timm.models.vision_transformer import vit_base_patch16_clip_224
from transformers import DebertaModel, DebertaConfig
import torch.nn.functional as F

from .mla import MultiLevelAdapter
from .d_tcr import DTCRBlock

class GeneralizedMeanPooling(nn.Module):
    """
    GeM Pooling as mentioned in Algorithm 1. 
    Provides a more robust global representation than standard Max or Average pooling.
    """
    def __init__(self, p=3, eps=1e-6):
        super().__init__()


    def forward(self, x):
        # x: (B, N, D) -> Pool over tokens (N)
  

class LDVQN(nn.Module):
    """
    Linguistically Driven Visual Query Network (LD-VQN).
    The core framework integrating hierarchical visual adaptation, dynamic routing, 
    and cross-modal alignment for fine-grained vehicle re-identification.
    
    Reference: Figure 6 and Section 3.2.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        embed_dim = 768 # Standard for ViT-B and DeBERTa-base
        
        # 1. Vision Backbone (ViT-B/16)
        # We use a version that allows extracting intermediate hidden states
        self.vision_encoder = vit_base_patch16_clip_224(pretrained=True, num_classes=0)
        
        # 2. Text Backbone (DeBERTa)
        # Pre-trained DeBERTa for capturing complex linguistic dependencies in 417K corpus
        self.text_encoder = DebertaModel.from_pretrained("microsoft/deberta-base")
        
        # 3. Multi-Level Adapter (MLA)
        # Anchor layers: [5, 11, 17, 23, final] as specified in configs/*.yml
        self.mla = MultiLevelAdapter(
     
        
        # 4. Dynamic Text-Conditioned Routing (D-TCR)
        self.dtcr = DTCRBlock(
            embed_dim=embed_dim,
         
        # 5. Global Aggregator and Neck
        self.gem_pool = GeneralizedMeanPooling()
        self.bottleneck = nn.BatchNorm1d(embed_dim)
 
        
        # 6. Classifier (Identity Prediction)
        num_classes = cfg['DATA'].get('NUM_CLASSES', 776) # Default to VeRi-776
   

    def forward(self, images, input_ids, attention_mask, labels=None):
        """
        Forward pass handling hierarchical extraction and dynamic routing.
        
        Args:
            images: (B, 3, 336, 336)


        Returns:
            Dict containing global features, logits, and component nodes (for Ortho Loss).
        """
        # Step 1: Hierarchical Vision Feature Extraction
        # We hook into the vision encoder to rescue early-stage primitives
        # anchor_indices: [5, 11, 17, 23, -1]
        anchor_indices = self.cfg['MODEL']['MLA']['ANCHOR_LAYERS']
        
        # Extract features through the transformer blocks
        x = self.vision_encoder.patch_embed(images)

        
        hidden_states = []
        for i, block in enumerate(self.vision_encoder.blocks):
            x = block(x)
            if i in anchor_indices or (i == len(self.vision_encoder.blocks)-1 and -1 in anchor_indices):
                hidden_states.append(x)
        
        # Step 2: MLA Fusion
        # f_img: (B, N_v, D)
        f_img = self.mla(hidden_states)
   
        
        if self.training:
            logits = self.classifier(f_bn)
            return {
                'global_feat': f_g,     # For Triplet Loss
                'logits': logits,       # For Cross-Entropy Loss


        else:
            # During inference, we use the BNNeck feature for retrieval
            return f_bn

    def load_param(self, model_path):
        """Utility for loading pre-trained weights during evaluation."""
        param_dict = torch.load(model_path, map_location='cpu')
        for i in param_dict:
            if 'classifier' in i: continue # Skip classifier for cross-domain transfer
    