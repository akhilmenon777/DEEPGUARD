import os
import shutil
import tempfile
import cv2
import numpy as np
import torch
import pytest

from backend.training.dataset import FaceRealFakeDataset
from backend.models.gend_vit import GenDStyleViT
from backend.training.train_image import train_image_model

@pytest.fixture
def dummy_140k_dir():
    temp_dir = tempfile.mkdtemp()
    # Mirror xhlulu/140k-real-and-fake-faces nested directory structure
    train_real = os.path.join(temp_dir, "real_vs_fake", "real-vs-fake", "train", "real")
    train_fake = os.path.join(temp_dir, "real_vs_fake", "real-vs-fake", "train", "fake")
    test_real = os.path.join(temp_dir, "real_vs_fake", "real-vs-fake", "test", "real")
    test_fake = os.path.join(temp_dir, "real_vs_fake", "real-vs-fake", "test", "fake")

    for d in [train_real, train_fake, test_real, test_fake]:
        os.makedirs(d, exist_ok=True)

    dummy_face = np.ones((256, 256, 3), dtype=np.uint8) * 128
    for i in range(4):
        cv2.imwrite(os.path.join(train_real, f"real_{i}.jpg"), dummy_face)
        cv2.imwrite(os.path.join(train_fake, f"fake_{i}.jpg"), dummy_face)
        cv2.imwrite(os.path.join(test_real, f"test_real_{i}.jpg"), dummy_face)
        cv2.imwrite(os.path.join(test_fake, f"test_fake_{i}.jpg"), dummy_face)

    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

def test_140k_dataset_loader(dummy_140k_dir):
    train_ds = FaceRealFakeDataset(root_dir=dummy_140k_dir, split="train", is_precropped=True)
    assert len(train_ds) == 8, f"Expected 8 training images, got {len(train_ds)}"

    tensor, label = train_ds[0]
    assert tensor.shape == (3, 224, 224)
    assert label.item() in [0, 1]

    # Test subsampling
    sub_ds = FaceRealFakeDataset(root_dir=dummy_140k_dir, split="train", max_samples=4)
    assert len(sub_ds) == 4

    # Test test split
    test_ds = FaceRealFakeDataset(root_dir=dummy_140k_dir, split="test", is_precropped=True)
    assert len(test_ds) == 8

def test_140k_training_step(dummy_140k_dir):
    train_ds = FaceRealFakeDataset(root_dir=dummy_140k_dir, split="train", is_precropped=True)
    temp_weights = os.path.join(dummy_140k_dir, "test_gend.pth")
    model = train_image_model(
        dataset=train_ds,
        epochs=1,
        batch_size=4,
        save_path=temp_weights,
        device="cpu"
    )
    assert isinstance(model, GenDStyleViT)
    assert os.path.exists(temp_weights)
