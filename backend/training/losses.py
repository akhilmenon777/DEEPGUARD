"""
Supervised Contrastive Loss for Hyperspherical Feature Manifold Learning.
Used in GenD to cluster genuine faces and separate manipulated faces in feature space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss (Khosla et al., NeurIPS 2020).
    Optimizes embeddings on the unit hypersphere: pulls samples of the same class together
    and pushes samples of different classes apart.
    """
    def __init__(self, temperature: float = 0.07, contrast_mode: str = "all"):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None, mask: torch.Tensor = None):
        """
        Args:
            features: Hidden vector of shape [batch_size, n_views, dim] or [batch_size, dim].
            labels: Ground truth labels of shape [batch_size].
            mask: Contrastive mask of shape [batch_size, batch_size], mask_{i,j}=1 if sample j
                  has the same class as sample i. Can be None.
        Returns:
            A scalar loss.
        """
        device = features.device

        if len(features.shape) == 2:
            # (batch_size, dim) -> unsqueeze to (batch_size, 1, dim)
            features = features.unsqueeze(1)

        batch_size = features.shape[0]

        if labels is not None and mask is None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        elif mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # Compute cosine similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T),
            self.temperature
        )

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask for multi-views
        mask = mask.repeat(contrast_count, contrast_count)
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * contrast_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log-probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # Loss
        loss = - (self.temperature / 0.07) * mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()

        return loss
