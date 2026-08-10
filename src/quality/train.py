"""Training script for the document quality CNN."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.quality.dataset import QualityDataset, train_transform, default_transform, IDX_TO_LABEL
from src.quality.model import QualityClassifier


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model. Returns loss, accuracy, all predictions and labels."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def train(
    labels_path: Path = Path("data/synthetic/labels.json"),
    synthetic_dir: Path = Path("data/synthetic"),
    output_dir: Path = Path("models/quality_cnn"),
    epochs_frozen: int = 5,
    epochs_finetune: int = 10,
    batch_size: int = 32,
    lr_frozen: float = 1e-3,
    lr_finetune: float = 1e-4,
    device: str = None,
):
    """Train the quality classifier in two phases.

    Phase 1: Frozen backbone — train only classifier head.
    Phase 2: Fine-tune — unfreeze backbone with lower learning rate.
    """
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else \
                 "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Datasets
    train_ds = QualityDataset(labels_path, synthetic_dir, transform=train_transform(), split="train")
    val_ds = QualityDataset(labels_path, synthetic_dir, transform=default_transform(), split="val")
    test_ds = QualityDataset(labels_path, synthetic_dir, transform=default_transform(), split="test")

    print(f"Dataset sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = QualityClassifier(num_classes=3, pretrained=True).to(device)

    # Class weights for imbalanced dataset
    label_counts = [0, 0, 0]
    for _, label in train_ds:
        label_counts[label] += 1
    total = sum(label_counts)
    class_weights = torch.tensor([total / (3 * c) for c in label_counts], dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights.cpu().tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Frozen backbone
    print(f"\n{'='*50}")
    print(f"Phase 1: Frozen backbone ({epochs_frozen} epochs, lr={lr_frozen})")
    print(f"{'='*50}")
    model.freeze_backbone()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_frozen)

    for epoch in range(epochs_frozen):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs_frozen} [{elapsed:.1f}s] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pth")

    # Phase 2: Fine-tune
    print(f"\n{'='*50}")
    print(f"Phase 2: Fine-tuning ({epochs_finetune} epochs, lr={lr_finetune})")
    print(f"{'='*50}")
    model.unfreeze_backbone()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_finetune)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    for epoch in range(epochs_finetune):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs_finetune} [{elapsed:.1f}s] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pth")

    # Test evaluation with best model
    print(f"\n{'='*50}")
    print("Test Set Evaluation (best model)")
    print(f"{'='*50}")
    model.load_state_dict(torch.load(output_dir / "best_model.pth", weights_only=True))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    target_names = [IDX_TO_LABEL[i] for i in range(3)]
    report = classification_report(labels, preds, target_names=target_names)
    cm = confusion_matrix(labels, preds)

    print(f"Test accuracy: {test_acc:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")

    # Save artifacts
    history["test_acc"] = test_acc
    history["test_loss"] = test_loss
    history["classification_report"] = classification_report(labels, preds, target_names=target_names, output_dict=True)
    history["confusion_matrix"] = cm.tolist()

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save(model.state_dict(), output_dir / "final_model.pth")
    print(f"\nArtifacts saved to {output_dir}/")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train quality classifier")
    parser.add_argument("--labels", type=Path, default=Path("data/synthetic/labels.json"))
    parser.add_argument("--synthetic-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/quality_cnn"))
    parser.add_argument("--epochs-frozen", type=int, default=5)
    parser.add_argument("--epochs-finetune", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-frozen", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(
        labels_path=args.labels,
        synthetic_dir=args.synthetic_dir,
        output_dir=args.output_dir,
        epochs_frozen=args.epochs_frozen,
        epochs_finetune=args.epochs_finetune,
        batch_size=args.batch_size,
        lr_frozen=args.lr_frozen,
        lr_finetune=args.lr_finetune,
        device=args.device,
    )
