"""
Unit tests for GenD Vision Transformer Model.
Verifies LayerNorm-only parameter freezing and hyperspherical projection.
"""
import torch
from backend.models.gend_vit import GenDStyleViT
from backend.training.losses import SupConLoss

def test_gend_layernorm_freezing():
    model = GenDStyleViT(model_name="vit_tiny_patch16_224", pretrained=False)
    summary = model.get_trainable_parameter_summary()

    # Total parameters should be non-zero
    assert summary["total_parameters"] > 0
    # Trainable parameters should be small fraction (< 15% even for tiny ViT, < 0.1% for base ViT)
    assert summary["trainable_parameters"] < summary["total_parameters"]

    # Verify that only LayerNorm / norm parameters have requires_grad == True in backbone
    for name, param in model.backbone.named_parameters():
        if "norm" in name.lower() or "ln" in name.lower():
            assert param.requires_grad is True, f"LayerNorm {name} should be trainable"
        else:
            assert param.requires_grad is False, f"Backbone layer {name} should be frozen"

def test_gend_forward_and_hypersphere():
    model = GenDStyleViT(model_name="vit_tiny_patch16_224", pretrained=False, proj_dim=128, num_classes=2)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    logits, embeddings = model(dummy_input)

    assert logits.shape == (2, 2)
    assert embeddings.shape == (2, 128)

    # Verify L2 hypersphere normalization: ||z||_2 should be 1.0
    norms = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "Embeddings must lie on unit hypersphere"

def test_supcon_loss():
    loss_fn = SupConLoss(temperature=0.07)
    features = F_norm = torch.nn.functional.normalize(torch.randn(4, 64), p=2, dim=1)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    loss = loss_fn(features, labels)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)
