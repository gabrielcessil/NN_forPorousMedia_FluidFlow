import torch.nn as nn
import torch
import json
from Utilities import loss_functions as lf
from Utilities import nn_trainner as nnt
from Utilities import model_handler as mh
from Utilities import loader_creator as lc
from Architectures import Inception_v3 as incp
from Utilities import dataset_reader as dr
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image




#######################################################
#************ USER INPUTS: Hyperparameters ***********#
#######################################################

batch_size = 10
max_samples = 3
train_ratio = 0.7
val_ratio = 0.15
learning_rate = 0.1
N_epochs = 3000
model_name = "Model" # The desired model name, avoid overwritting previous models
loss_functions = {
    "MSE":  {"obj":     nn.MSELoss(),"Thresholded": False},
    "L1":   {"obj":     nn.L1Loss(), "Thresholded": False},
    "Complementary MIOU": {"obj":     lf.CustomLoss_MIOU(),"Thresholded": True},
    "Complementary Accuracy": {"obj": lf.CustomLoss_Accuracy(),"Thresholded": True}
    }
earlyStopping_loss = "Complementary Accuracy"
backPropagation_loss = "MSE"



#######################################################
#************ LOADING CONFIGS ************************#
#######################################################

# LOADING CONFIGS
with open("config.json" , "r") as json_file:
    config_loaded = json.load(json_file)
examples_shape = config_loaded["Rock_shape"]
NN_dataset_folder = config_loaded["NN_dataset_folder"]
NN_model_weights_folder = config_loaded["NN_model_weights_folder"]
model_full_name = NN_model_weights_folder+model_name



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
        
# Segmentation Benchmarks:https://paperswithcode.com/task/semantic-segmentation
# Original content: https://project.inria.fr/aerialimagelabeling/files/
# Dataset's Benchmark: https://paperswithcode.com/sota/semantic-segmentation-on-inria-aerial-image

## Example 1: FOREST SEGMENTATION
"""
#import kagglehub
#path = kagglehub.dataset_download("quadeer15sh/augmented-forest-segmentation")
#print("Path to dataset files:", path)
path = "/home/gabriel/.cache/kagglehub/datasets/quadeer15sh/augmented-forest-segmentation/versions/2/"
dataset_path = path+"/Forest Segmented/Forest Segmented/"
examples_shape = [3, 256, 256]
output_shape = [1, 256, 256]
data = ForestSegmentationData(dataset_path, examples_shape)
def make_binary_int(a): return (a > 0.5).int() # Make it binary
output_masks = [make_binary_int]
"""


## Example 2: INRIA SEGMENTATION
dataset_path = "/home/gabriel/Downloads/AerialLabelling/AerialImageDataset_01/train/"
examples_shape = [3, 500, 500]
output_shape = [1, 500, 500]
data = AerialSegmentationData(dataset_path, examples_shape)
def make_binary_int(a): return (a > 0).int() # Make it binary
output_masks = [make_binary_int]




#######################################################
#************ COMPUTATIONS ***************************#
#######################################################d

# CREATING DATALOADER
input_tensors = data.inputs
output_tensors = data.outputs

dataset = lc.Data_Loader(input_tensors, output_tensors)
train_loader, val_loader, test_loader = dataset.get_splitted(train_ratio=train_ratio,
                                                            val_ratio=val_ratio,
                                                            batch_size=batch_size,
                                                            max_samples=max_samples)

# Print dataset statistics
print(" Dataset Split Summary:")
print(f"  Total samples: {input_tensors.shape[0]}")  # Number of samples (N)
print(f"  - Total Train samples: {len(train_loader.dataset)}")
print(f"  - Total Validation samples: {len(val_loader.dataset)}")
print(f"  - Total Test samples: {len(test_loader.dataset)}")
print(f"  Number of training batches: {len(train_loader)}")

# Fetch a batch and print correct shapes
batch_inputs, batch_outputs = next(iter(train_loader))
input_shape = batch_inputs[0].shape
output_shape = batch_outputs[0].shape

print("\n Batch Shape Details: (batch_size, channels, height, weight)")
print(f" - Train Batch Input Shape: {input_shape}")  # (batch_size, C, H, W)
print(f" - Train Batch Output Shape: {output_shape}")  # (batch_size, ...)
print(f" - Training samples per batch: {batch_inputs.shape[0]}")  # Correct batch size




# CREATING MODEL
model = incp.MODEL(
    in_shape=input_shape,  # (C, H, W)
    out_shape=output_shape,  # (C, H, W)
    output_masks=output_masks
)


# RUNNING EXAMPLE WITHOUT TRAINNING
nnt.Run_Example(model, train_loader,loss_functions, earlyStopping_loss, i_example=0, title="Train sample (without trainning)")
nnt.Run_Example(model, test_loader,loss_functions, earlyStopping_loss, i_example=0, title="Test sample (without trainning)")


#### MODEL TRAINNING
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

model_paths, train_costs_history, val_costs_history = nnt.full_train(
                                                            model,
                                                            train_loader,
                                                            val_loader,                    
                                                            loss_functions,                                      
                                                            earlyStopping_loss,
                                                            backPropagation_loss,
                                                            optimizer,
                                                            N_epochs=N_epochs,
                                                            file_name=model_full_name)

nnt.Plot_Validation(train_costs_history, val_costs_history)



#### MODEL TEST
model_path = model_paths[0]
model.load_state_dict(torch.load(model_path)) # Load model 
nnt.Run_Example(model, train_loader,loss_functions, earlyStopping_loss, i_example=0, title="Train sample")
nnt.Run_Example(model, test_loader,loss_functions, earlyStopping_loss, i_example=0, title="Test sample")

### DELETE MODEL AFTER USING IT
mh.delete_model(model)
