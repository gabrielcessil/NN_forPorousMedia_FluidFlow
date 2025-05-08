import torch.nn as nn
from torchmetrics.classification import Accuracy
from torchmetrics.segmentation import MeanIoU
from torch.nn import BCELoss
import torch


#######################################################
#************ LOSS FUNCTION UTILITIES ****************#
#######################################################

# Apply threshold
class Binarize(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        return (torch.sigmoid(x) > self.threshold).float()
    
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
    def _default_mask_law(self,output, target, threshold=0.0001): 
        
        # Mask consider only target != 0, i.e, non-solid cells
        mask = (target > threshold) | (target < -threshold)
        return mask
    
    def forward(self, output, target):
        if output.size() != target.size():
            raise ValueError(f"CustomLoss forward: Tensors have different sizes ({output.size()} vs {target.size()})")
        mask = self.mask_law(output, target)
        
        target = target[mask]
        output = output[mask]

        return self.lossFunction(output, target)
    
    
    
    
#######################################################
#************ LOSS FUNCTIONS  ************************#
#######################################################


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

class CustomLoss_IOU(nn.Module):
    
    # Based on: Using Intersection over Union loss to improve Binary Image Segmentation
    # Link: https://fse.studenttheses.ub.rug.nl/18139/1/AI_BA_2018_FlorisvanBeers.pdf
    # Resource: https://www.youtube.com/watch?v=NqDBvUPD9jg
    def __init__(self):
        super(CustomLoss_IOU, self).__init__()
        
    def forward(self, output, target):
        if output.size() != target.size():
            raise ValueError(f"CustomLoss_MIOU forward: Tensors have different sizes ({output.size()} vs {target.size()})")
        
        
        # Flatten the tensors
        T = target.flatten().float()
        P = output.flatten().float()

        # Ensure both tensors are floating point type for accurate calculations
        T = T.float()
        P = P.float()

        # Calculate the intersection
        intersection = torch.sum(T * P)
        union = torch.sum(T) + torch.sum(P) - intersection

        # Calculate the IOU
        result = (intersection + 1.0) / (union + 1.0)

        return 1 - result


    

        
