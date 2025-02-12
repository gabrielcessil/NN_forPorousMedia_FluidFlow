import torch
import numpy as np

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
        dataset = torch.load(self.file_path)
        
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
