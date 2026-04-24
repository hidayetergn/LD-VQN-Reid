import torch
import torch.nn as nn
from transformers import DebertaModel, DebertaConfig

class DebertaTextEncoder(nn.Module):
    """
    Text Encoder wrapper for Microsoft's DeBERTa model.
    Extracts deep linguistic features from the 417K unconstrained natural language captions.
    
    Implements the shallow-layer freezing strategy defined in Algorithm 1 to optimize 
    memory footprint and prevent catastrophic forgetting during cross-modal alignment.
    """
    def __init__(self, model_name="microsoft/deberta-base", freeze_layers=4):
        """
        Args:
            model_name (str): HuggingFace model identifier.
            freeze_layers (int): Number of initial transformer layers to freeze.
                                 Setting this to >0 drastically reduces VRAM usage
                                 while maintaining semantic extraction capabilities.
        """
        super().__init__()
        self.model_name = model_name
        
        # Load the pre-trained DeBERTa configuration and model
        self.config = DebertaConfig.from_pretrained(model_name)
 
        
        # Apply the freezing strategy (Algorithm 1)
        self._freeze_shallow_layers(freeze_layers)

    def _freeze_shallow_layers(self, num_layers_to_freeze):
        """
        Freezes the word embeddings and the first N transformer layers.
        """
        if num_layers_to_freeze <= 0:
            return

        # 1. Freeze raw word/position embeddings
        for param in self.encoder.embeddings.parameters():

            
        # 2. Freeze the specified number of shallow encoder layers
        for i in range(min(num_layers_to_freeze, len(self.encoder.encoder.layer))):

        print(f"[*] DeBERTa Text Encoder: Froze embeddings and first {num_layers_to_freeze} layers.")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tokenized text indices (B, N_t)

        Returns:
            last_hidden_state: Deep linguistic token embeddings (B, N_t, D)
        """
        # Forward pass through DeBERTa
        outputs = self.encoder(
            input_ids=input_ids,
  
        
        # We extract the sequence of hidden states for the D-TCR module
        # Shape: (B, N_t, 768)
        last_hidden_state = outputs.last_hidden_state


    @property
    def feature_dim(self):
        """Returns the output feature dimension."""
