import os
import json
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch

# Prevent PIL from crashing on truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class VeRi776Dataset(Dataset):
    """
    Multimodal Dataset class for the semantically enriched VeRi-776.
    Reads .jsonl files containing {"filename", "id", "caption"}.
    """
    def __init__(self, jsonl_path, img_dir, transform=None, tokenizer=None, max_seq_length=128):
        """
        Args:
            jsonl_path (str): Path to the .jsonl annotation file.
            img_dir (str): Root directory containing the images.
            transform (callable, optional): Torchvision transforms for the image.
            tokenizer (callable, optional): DeBERTa tokenizer instance.
            max_seq_length (int): Maximum token sequence length.
        """
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = self._load_jsonl(jsonl_path)

    def _load_jsonl(self, filepath):
        """Safely loads annotations into memory."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Annotation file not found: {filepath}")
        
        data_list = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                data_list.append(record)
        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        img_name = record['filename']
        pid = int(record['id'])
        caption = record['caption']

        # 1. Load and process image
        img_path = os.path.join(self.img_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Error loading image {img_path}: {e}")

        if self.transform is not None:
            img = self.transform(img)

        # 2. Tokenize text (if tokenizer is provided)
        text_inputs = {}
        if self.tokenizer is not None:
            text_inputs = self.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            # Squeeze batch dimension added by tokenizer
            text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}

        return {
            'image': img,
            'input_ids': text_inputs.get('input_ids', torch.empty(0)),
            'attention_mask': text_inputs.get('attention_mask', torch.empty(0)),
            'pid': pid,
            'caption': caption, # Kept for qualitative visualizations (Fig. 9)
            'img_path': img_path
        }