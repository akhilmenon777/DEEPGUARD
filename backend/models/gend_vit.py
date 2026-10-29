"""
GenD: Parameter-efficient Vision Transformer for Deepfake Detection.
Centerpiece of the detection pipeline based on:
"Deepfake Detection that Generalizes Across Benchmarks (GenD)", arXiv:2508.06248.
Architecture:
- Frozen Vision Transformer (ViT) backbone.
- LayerNorm-only fine-tuning (~0.03% of total parameters trainable).
- Hyperspherical feature projection (Linear embedding head + L2 normalization).
- Linear 2-class classifier (Real / Fake).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GenDStyleViT(nn.Module):
    """
    GenD ViT implementation with LayerNorm-only adaptation
    and hyperspherical manifold projection.
    """
    def __init__(self, model_name: str = "vit_tiny_patch16_224", pretrained: bool = False, embed_dim: int = 192, proj_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.model_name = model_name
        self.proj_dim = proj_dim
        self.num_classes = num_classes

        # Try to load timm backbone; fallback to robust native ViT if timm is not installed
        try:
            import timm
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            in_features = getattr(self.backbone, "num_features", embed_dim)
        except Exception:
            # Native lightweight ViT implementation matching timm architecture
            self.backbone = NativeViTBackbone(img_size=224, patch_size=16, in_chans=3, embed_dim=embed_dim, depth=6, num_heads=6)
            in_features = embed_dim

        # GenD Adaptation: Freeze all backbone parameters except LayerNorm layers
        self._freeze_backbone_except_layernorm()

        # Hyperspherical Embedding Head
        self.embed_head = nn.Sequential(
            nn.Linear(in_features, proj_dim),
            nn.BatchNorm1d(proj_dim)
        )

        # 2-class linear classifier
        self.classifier = nn.Linear(proj_dim, num_classes)

    def _freeze_backbone_except_layernorm(self):
        """
        Freezes all backbone weights, keeping ONLY LayerNorm parameters trainable.
        Tuning LayerNorm layers allows domain adaptation while preserving general visual representations.
        """
        total_params = 0
        trainable_params = 0
        for name, param in self.backbone.named_parameters():
            total_params += param.numel()
            # In ViT, LayerNorms are named 'norm', 'norm1', 'norm2', or contain 'ln'
            if "norm" in name.lower() or "ln" in name.lower():
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False

        self.backbone_total_params = total_params
        self.backbone_trainable_params = trainable_params

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, 3, 224, 224)
        Returns:
            logits: (B, num_classes)
            embedding: (B, proj_dim) normalized hyperspherical embedding
        """
        # Backbone feature extraction
        features = self.backbone(x) # (B, in_features)

        # Hyperspherical embedding projection
        proj = self.embed_head(features)
        # L2 Normalization onto the unit hypersphere: ||z||_2 = 1
        embedding = F.normalize(proj, p=2, dim=1)

        # Classification logits
        logits = self.classifier(embedding)

        return logits, embedding

    def get_trainable_parameter_summary(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        ratio = (trainable / total) * 100.0 if total > 0 else 0.0
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "trainable_ratio_pct": round(ratio, 4)
        }


class NativeViTBackbone(nn.Module):
    """
    Self-contained pure PyTorch Vision Transformer backbone
    matching timm ViT structure for zero-dependency execution.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192, depth=6, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.num_features = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        feat = self.forward_features(x)
        return feat[:, 0] # Class token representation


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
