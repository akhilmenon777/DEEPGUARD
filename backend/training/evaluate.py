"""
Robustness and In-The-Wild Generalization Evaluation Module.
Implements the evaluation framework described in project analysis:
Measures accuracy on source dataset vs. perturbed in-the-wild media
and computes the cross-dataset Generalization Gap (Deepfake-Eval-2024 style).
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple

from backend.models.gend_vit import GenDStyleViT
from backend.training.dataset import FaceRealFakeDataset

def generalisation_gap(source_acc: float, wild_acc: float) -> float:
    """
    Computes the generalization drop:
    Gap = Accuracy(Source) - Accuracy(In-The-Wild)
    A smaller gap indicates better cross-dataset robustness (the core goal of GenD).
    """
    return round(float(source_acc - wild_acc), 2)

def evaluate_image_folder(model: GenDStyleViT, dataset: FaceRealFakeDataset, device: str = "cpu", batch_size: int = 8) -> Dict[str, Any]:
    """
    Evaluates GenDStyleViT across a dataset of real and fake face images.
    Returns accuracy, false positive rate, false negative rate, and average fake probability.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    tp = 0 # True Fake
    fp = 0 # False Fake (real classified as fake)
    tn = 0 # True Real
    fn = 0 # False Real (fake classified as real)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits, _ = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += len(labels)

            for p, pred, gt in zip(probs.cpu().numpy(), preds.cpu().numpy(), labels.cpu().numpy()):
                all_probs.append(float(p))
                all_labels.append(int(gt))
                if gt == 1 and pred == 1:
                    tp += 1
                elif gt == 0 and pred == 1:
                    fp += 1
                elif gt == 0 and pred == 0:
                    tn += 1
                elif gt == 1 and pred == 0:
                    fn += 1

    accuracy = (correct / max(1, total)) * 100.0
    precision = (tp / max(1, tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    recall = (tp / max(1, tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / max(1e-6, precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "total_samples": total,
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "mean_manipulation_score": round(float(np.mean(all_probs)), 4) if all_probs else 0.0
    }

def run_robustness_benchmark(
    model: GenDStyleViT,
    source_dataset: FaceRealFakeDataset,
    in_the_wild_dataset: FaceRealFakeDataset,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Executes comparative evaluation across controlled source distribution
    and real-world in-the-wild media stream.
    """
    source_metrics = evaluate_image_folder(model, source_dataset, device=device)
    wild_metrics = evaluate_image_folder(model, in_the_wild_dataset, device=device)
    gap = generalisation_gap(source_metrics["accuracy"], wild_metrics["accuracy"])

    return {
        "source_dataset_evaluation": source_metrics,
        "in_the_wild_evaluation": wild_metrics,
        "generalisation_gap_pct": gap,
        "status": "Robust GenD Generalization" if gap < 15.0 else "Susceptible to Real-World Distribution Shift"
    }
