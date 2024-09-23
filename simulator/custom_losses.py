import torch
import torch.nn as nn
import torch.nn.functional as F


class CDWCELoss(nn.Module):
    def __init__(self, alpha=5):
        super(CDWCELoss, self).__init__()
        self.alpha = alpha

    def forward(self, predictions, targets):
        targets = torch.argmax(targets, dim=1)
        # Get the number of classes
        num_classes = predictions.size(1)

        # Create a range tensor for class indices
        class_indices = torch.arange(num_classes, device=predictions.device)

        # Expand dimensions for broadcasting
        targets_expanded = targets.unsqueeze(1)
        class_indices_expanded = class_indices.unsqueeze(0)

        # Calculate the distance between predicted classes and true class
        distances = torch.abs(class_indices_expanded - targets_expanded)

        # Apply the distance weighting
        weights = torch.pow(distances, self.alpha)

        # Calculate log(1 - predicted_prob) for each class in a numerically stable way
        log_probs = F.logsigmoid(-predictions)

        # Apply the weights to the log probabilities
        weighted_log_probs = weights * log_probs

        # Sum over all classes
        loss = -torch.sum(weighted_log_probs, dim=1)

        # Return the mean loss
        return torch.mean(loss)
