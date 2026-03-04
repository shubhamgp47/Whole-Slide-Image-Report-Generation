import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import PathologyTitanEmbeddingDataset

class TitanR2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        # REMOVED: Image transforms (not needed for pre-computed embeddings)
        # No self.transform since we're loading embeddings directly
        
        if self.dataset_name == 'wsi_report':
            self.dataset = PathologyTitanEmbeddingDataset(self.args, self.tokenizer, self.split)
        else:
            raise ValueError

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        """Modified collate function for TITAN embeddings"""
        # Data now contains: (image_ids, slide_embeddings, reports_ids, reports_masks, seq_lengths)
        images_id, slide_embeddings, reports_ids, reports_masks, seq_lengths = zip(*data)
        
        # Stack slide embeddings instead of images
        slide_embeddings = torch.stack(slide_embeddings, 0)  # [batch_size, titan_embedding_dim]
        
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        # Return slide_embeddings instead of images
        return images_id, slide_embeddings, torch.LongTensor(targets), torch.FloatTensor(targets_masks)

    

    #Added my function to pad the smaller patches. 
    # RuntimeError: stack expects each tensor to be equal size, but got [384, 1024] at entry 0 and [928, 1024] at entry 1
    '''@staticmethod
    def collate_fn(data):
        # Unpack batch data
        images_id, images, reports_ids, reports_masks, seq_lengths = zip(*data)  # data is a list of tuples

        # Find max patch count in batch (images are [patches, feature_dim])
        max_patches = max(img.shape[0] for img in images)
        feature_dim = images[0].shape[1]

        # Prepare padded tensor for images and mask for valid patches
        batch_size = len(images)
        padded_images = torch.zeros(batch_size, max_patches, feature_dim)  # zero padding
        patches_mask = torch.zeros(batch_size, max_patches, dtype=torch.bool)  # mask: True where valid patch

        for i, img in enumerate(images):
            patch_count = img.shape[0]
            padded_images[i, :patch_count] = img
            patches_mask[i, :patch_count] = 1  # valid patches

        # Process reports (same as before)
        max_seq_length = max(seq_lengths)
        targets = np.zeros((batch_size, max_seq_length), dtype=int)
        targets_masks = np.zeros((batch_size, max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_mask in enumerate(reports_masks):
            targets_masks[i, :len(report_mask)] = report_mask

        return images_id, padded_images, patches_mask, torch.LongTensor(targets), torch.FloatTensor(targets_masks)'''


class CompetitionDataLoaderForTest(DataLoader):
    """
    Alternative DataLoader specifically for competition submissions
    Always uses CompetitionPathologyDataset and returns (image_id, image) format
    """
    def __init__(self, args, tokenizer, split='test', shuffle=False):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        # Always use test transforms for competition
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        if self.dataset_name == 'wsi_report':
            self.dataset = CompetitionPathologyDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            raise ValueError

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.competition_collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def competition_collate_fn(data):
        """Collate function for competition data (image_id, image)"""
        images_id, images = zip(*data)
        images = torch.stack(images, 0)
        return images_id, images