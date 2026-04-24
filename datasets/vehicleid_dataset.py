import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torch

class VehicleIDDataset(Dataset):
    """
    Multimodal Dataset class for the semantically enriched VehicleID.
    Supports dynamic loading of progressive test splits (800, 1600, 3200, etc.).
    """
    def __init__(self, jsonl_path, img_dir, transform=None, tokenizer=None, max_seq_length=128):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = self._load_jsonl(jsonl_path)

    def _load_jsonl(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Annotation file not found: {filepath}")
        
        data_list = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # VehicleID JSONL structure: {"filename": "...", "id": "...", "caption": "..."}
                data_list.append(json.loads(line.strip()))
        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        img_name = record['filename']
        # Depending on formatting, ID might be a string in VehicleID. Ensure integer map.
        pid = int(record['id']) 
        caption = record.get('caption', "")

        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        text_inputs = {}
        if self.tokenizer is not None and caption:
            text_inputs = self.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}

        return {
            'image': img,
            'input_ids': text_inputs.get('input_ids', torch.empty(0)),
            'attention_mask': text_inputs.get('attention_mask', torch.empty(0)),
            'pid': pid,
            'img_path': img_path
        }