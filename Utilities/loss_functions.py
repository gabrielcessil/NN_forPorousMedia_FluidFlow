import torch.nn as nn
from torchmetrics.classification import Accuracy
from torchmetrics.segmentation import MeanIoU
from torch.nn import BCELoss
import torch

class Binarize(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        return (torch.sigmoid(x) > self.threshold).float()
    
    
class CustomLoss_Accuracy(nn.Module):
    def __init__(self):
        super(CustomLoss_Accuracy, self).__init__()
        self.accuracy = Accuracy(task="binary")  # Use "multiclass" for multiple classes

    def forward(self, output, target):
        if output.size() != target.size():
            raise ValueError(f"CustomLoss_Accuracy forward: Tensors have different sizes ({output.size()} vs {target.size()})")

        # Convert probabilities/logits to binary values
        preds = (output > 0.5).int()
        target = target.int()  # Ensure target is also in integer format

        acc = self.accuracy(preds, target)
        return 1 - acc  # Convert accuracy to a loss (lower is better)
    

class Custom_BCE(nn.Module):
    def __init__(self):
        super(Custom_BCE, self).__init__()
        self.bce = BCELoss()
        
    def forward(self,output, target):
        if output.size() != target.size():
            raise ValueError(f"CustomLoss_Accuracy forward: Tensors have different sizes ({output.size()} vs {target.size()})")

        # Ensure float type
        output = output.float()
        target = target.float()
        
        # Clamp output to avoid log(0) issues
        output = torch.clamp(output, min=1e-7, max=1 - 1e-7)
        
        return self.bce(output, target)
    
class CustomLoss_MIOU(nn.Module):
    def __init__(self):
        super(CustomLoss_MIOU, self).__init__()
        self.miou = MeanIoU(num_classes=2)
        
    def forward(self, output, target):
        if output.size() != target.size():
            raise ValueError(f"CustomLoss_MIOU forward: Tensors have different sizes ({output.size()} vs {target.size()})")
        
        target = (target>0.5).int()
        
        self.miou.update(output, target)
        result = self.miou.compute()
        self.miou.reset()
        return  1 - result


# Defines the specific pixels that the loss function might look into
class Mask_LossFunction(nn.Module):
    def __init__(self, lossFunction, mask_law=None):
        super(Mask_LossFunction, self).__init__()
        
        self.lossFunction = lossFunction
        
        if mask_law is None: 
            self.mask_law = self._default_mask_law
        else:
            self.mask_law = mask_law
            
    # Do not consider cells with 0 value  
    # The loss function used must be a mean across the tensor lenght, 
    # so that the quantity of solid cells do not affect the loss
    def _default_mask_law(self,output, target): 
        mask = target != 0
        return mask
    
    def forward(self, output, target):
        if output.size() != target.size():
            raise ValueError(f"CustomLoss forward: Tensors have different sizes ({output.size()} vs {target.size()})")
        mask = self.mask_law(output, target)
        
        target = target[mask]
        output = output[mask]

        return self.lossFunction(output, target)
    

        
