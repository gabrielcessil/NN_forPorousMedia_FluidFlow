import torch.nn as nn
from torchmetrics.classification import Accuracy
from torchmetrics.segmentation import MeanIoU

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

        # Compute accuracy
        acc = self.accuracy(preds, target)
        
        return 1 - acc  # Convert accuracy to a loss (lower is better)
    

class CustomLoss_MIOU(nn.Module):
    def __init__(self):
        super(CustomLoss_MIOU, self).__init__()

    def forward(self, output, target):
        if output.size() != target.size():
            raise ValueError(f"CustomLoss_MIOU forward: Tensors have different sizes ({output.size()} vs {target.size()})")
        
        target = (target>0.5).int()
        miou = MeanIoU(num_classes=2)
        miou.update(output, target)
        result = miou.compute()
        miou.reset()
        return  1 - result

class Mask_LossFunction(nn.Module):
    def __init__(self, lossFunction, mask_law=None):
        super(Mask_LossFunction, self).__init__()
        
        self.lossFunction = lossFunction
        
        if mask_law is None: 
            self.mask_law = self._default_mask_law
        else:
            self.mask_law = mask_law
        
    def forward(self, output, target):
        if output.size() != target.size():
            raise ValueError(f"CustomLoss forward: Tensors have different sizes ({output.size()} vs {target.size()})")
        
        # Remove solid cells (=0). 
        # The loss function used must be a mean across the tensor lenght, 
        # so that the quantity of solid cells do not affect the loss
        
        mask = self.mask_law(target)
        
        target = target[mask]
        output = output[mask]

        return self.lossFunction(output, target)
    
    def _default_mask_law(self,target): 
        mask = target != 0
        return mask
        
