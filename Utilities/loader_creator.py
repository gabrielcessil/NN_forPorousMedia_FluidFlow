import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import Dataset

class Data_Loader(Dataset):
    def __init__(self, inputs_tensor, outputs_tensor):
        
        self.inputs, self.outputs = inputs_tensor, outputs_tensor

    def get_splitted(self, train_ratio=0.7, val_ratio=0.15, batch_size=10, max_samples=None):
        train_dataset, val_dataset, test_dataset = self._split(
            train_ratio=train_ratio, val_ratio=val_ratio, max_samples=max_samples
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)  # No shuffle for val/test
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
    
    def _split(self, train_ratio=0.7, val_ratio=0.15, max_samples=None):
        total_size = len(self)
        if total_size <= 3:
            raise Exception("TensorsDataset must have at least 3 samples in order to split")
    
        if max_samples is not None:
            total_size = min(total_size, max_samples)  # Limit dataset size
    
        # Compute sizes ensuring at least 1 sample per dataset
        train_size = max(1, int(train_ratio * total_size))
        val_size = max(1, int(val_ratio * total_size))
        
        # Ensure remaining samples go to the test set
        test_size = max(1, total_size - (train_size + val_size))  
    
        # Adjust if rounding errors cause an overflow
        if train_size + val_size + test_size > total_size:
            train_size = total_size - (val_size + test_size)
    
        # Generate indices and split dataset
        indices = list(range(total_size))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
    
        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)
        test_dataset = Subset(self, test_indices)
    
        return train_dataset, val_dataset, test_dataset
    
    
    