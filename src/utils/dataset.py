# -*- coding: utf-8 -*-

from torch.utils.data import Dataset

class ResumeDataset(Dataset):
    """Dataset class for resume texts."""
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]