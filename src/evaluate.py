from __future__ import annotations

import argparse
import torch
from torch import nn

from dataset import get_mnist_loaders
from models import build_model
from utils import get_device, load_checkpoint, set_seed


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MNIST model.")
    parser.add_argument("--model", choices=["resnet", "vgg"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    _, test_loader = get_mnist_loaders(args.data_dir, args.batch_size, args.num_workers)
    model = build_model(args.model).to(device)
    load_checkpoint(model, args.checkpoint, device)
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
