import torch
import numpy as np
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image

class LBM_Dataset_Processor:
    """
    Dataset Loader & Preprocessor

    Loads a `.pt` dataset, handling folder structures and various formats, ensuring 
    correct conversion to PyTorch tensors for model training.

    Features:
    - Reads `.pt` files and processes inputs/outputs.
    - Reshapes arrays as needed and converts to tensors.
    - Handles lists, NumPy arrays, and different data structures.
    - Stores tensors as `self.inputs` and `self.outputs`.

    Usage:
    ```
    processor = LBM_Dataset_Processor("your_dataset.pt", array_shape=(50, 50))
    inputs, outputs = processor.get_tensors()
    ```
    """

    def __init__(self, file_path):
        """
        Initializes the dataset processor by loading and processing data.

        Args:
            file_path (str): Path to the `.pt` dataset file.
            array_shape (tuple): Expected shape for input images (e.g., (50, 50)).
        """
        self.file_path = file_path
        
        self.inputs, self.outputs = self.load_data()

    def array_to_numpy(self, variable):
        """Ensures the input is a NumPy array."""
        if isinstance(variable, np.ndarray):
            return variable
        elif isinstance(variable, list):
            return np.array(variable)
        raise TypeError("Dataset samples must be a list or a numpy ndarray.")

    def reshape_array(self, array):
        """Reshapes an array to the expected shape if necessary."""
        if not isinstance(array, np.ndarray):
            raise TypeError("Input must be a numpy ndarray.")
        if array.shape == self.array_shape:
            return array  # Already correct shape
        try:
            return array.reshape(self.array_shape)
        except ValueError:
            raise ValueError(f"Cannot reshape array of shape {array.shape} to {self.array_shape}")

    def convert_to_tensor(self, array, flatten=False, add_channel_dim=False):
        """Converts a NumPy array to a PyTorch tensor with optional modifications."""
        tensor = torch.tensor(array, dtype=torch.float32)
        if flatten:
            tensor = tensor.view(-1)  # Flatten
        if add_channel_dim:
            tensor = tensor.unsqueeze(0)  # Add channel dim (for Conv2D input)
        return tensor

    def load_data(self):
        """Loads and processes the dataset into tensors."""
        dataset = torch.load(self.file_path, weights_only=False)
        
        inputs = []
        outputs = []
        
        # Convert pairs of arrays into tensors
        for pair in dataset:
            if len(pair) != 2:
                raise ValueError("Each dataset sample must contain 2 elements (input and ouput).")
            
            # Store tensors
            inputs.append(self.convert_to_tensor(pair[0]))
            outputs.append(self.convert_to_tensor(pair[1]))  # Stack outputs together

        # Convert lists to tensors
        inputs = torch.stack(inputs)
        outputs = torch.stack(outputs)
        
        return inputs, outputs

    def get_tensors(self):
        """Returns the processed input and output tensors."""
        return self.inputs, self.outputs


#######################################################
#**** CUSTOMIZATION TO DEAL WITH DATASET STRUCTURE ***#
#######################################################
from torchvision import transforms

class ForestSegmentationData(Dataset):

    def __init__(self, dataset_path, examples_shape):
        csv_file = os.path.join(dataset_path, "meta_data.csv")
        img_dir = os.path.join(dataset_path, "images")
        mask_dir = os.path.join(dataset_path, "masks")

        transform = transforms.Compose([
            transforms.Resize((examples_shape[1],examples_shape[2])),  # Padronizar tamanho
            transforms.ToTensor() # Converter para tensor
        ])
        
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.inputs = []
        self.outputs = []
        self.load_data()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

    def load_data(self):
        for idx in range(len(self.data)):
            img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
            mask_path = os.path.join(self.mask_dir, self.data.iloc[idx, 1])

            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)

            self.inputs.append(image)
            self.outputs.append(mask)

        self.inputs = torch.stack(self.inputs)
        self.outputs = torch.stack(self.outputs)
    

class AerialSegmentationData(Dataset):
    def __init__(self, dataset_path, examples_shape):
        img_dir = os.path.join(dataset_path, "images")
        mask_dir = os.path.join(dataset_path, "gt")

        self.transform = transforms.Compose([
            transforms.Resize((examples_shape[1], examples_shape[2])),  
            transforms.ToTensor()
        ])

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.inputs = []
        self.outputs = []
        self.load_data()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

    def load_data(self):
        img_files = sorted(os.listdir(self.img_dir))
        mask_files = sorted(os.listdir(self.mask_dir))

        assert len(img_files) == len(mask_files), "Mismatch between images and masks"

        for img_file, mask_file in zip(img_files, mask_files):
            img_path = os.path.join(self.img_dir, img_file)
            mask_path = os.path.join(self.mask_dir, mask_file)

            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)

            self.inputs.append(image)
            self.outputs.append(mask)

        self.inputs = torch.stack(self.inputs)
        self.outputs = torch.stack(self.outputs)