import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import h5py
import numpy as np

class BaseTitanDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.slide_embedding_dir = args.slide_embedding_dir  # Changed from image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.embedding_format = args.embedding_format  # New parameter for embedding format
        # transform parameter kept for compatibility but not used
        
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            # Below is the code to generate the mask for the report
            # such a mask is used to indicate the positions of actual tokens versus padding positions in a sequence.
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)

class PathologyTitanEmbeddingDataset(BaseTitanDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        
        # Load pre-computed TITAN slide embedding
        embedding_path = os.path.join(self.slide_embedding_dir, image_id + f'_slide_embedding.{self.embedding_format}')
        
        # Load embedding based on format
        if self.embedding_format == 'pt':
            slide_embedding = torch.load(embedding_path)
        elif self.embedding_format == 'npy':
            slide_embedding = torch.from_numpy(np.load(embedding_path)).float()
        elif self.embedding_format == 'h5':
            with h5py.File(embedding_path, 'r') as f:
                slide_embedding = torch.from_numpy(f['slide_embedding'][:]).float()
        else:
            raise ValueError(f"Unsupported embedding format: {self.embedding_format}")
        
        # FIX: Convert to float32 and squeeze out extra dimensions
        slide_embedding = slide_embedding.float().squeeze()  # Remove extra dimensions
        
        # Ensure it's 1D with correct shape [768]
        if slide_embedding.dim() == 0:
            slide_embedding = slide_embedding.unsqueeze(0)
        
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        
        return image_id, slide_embedding, report_ids, report_masks, seq_length

