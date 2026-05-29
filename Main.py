
# 1. Imports
import os
import random
import argparse
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


# 2. Config Defaults
TASK_NAMES = {
    0: "Is even?",
    1: "Is odd?",
    2: "Greater than 5?",
    3: "Less than 3?"
}


# 3. Utilities
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# 4. Dataset
class ExpandedMNISTTaskDataset(Dataset):
    """
    Converts MNIST into a multimodal task dataset.

    Original MNIST gives:
        image, digit_label

    This dataset gives:
        image, task_id, binary_label, digit_label

    Each MNIST image appears with all 4 task IDs:
        task 0: Is the digit even?
        task 1: Is the digit odd?
        task 2: Is the digit greater than 5?
        task 3: Is the digit less than 3?
    """

    def __init__(self, root: str = "./data", train: bool = True, transform=None):
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transform
        )

        self.num_tasks = 4

    def __len__(self):
        return len(self.mnist) * self.num_tasks

    def __getitem__(self, idx):
        mnist_idx = idx // self.num_tasks
        task_id = idx % self.num_tasks

        image, digit_label = self.mnist[mnist_idx]

        binary_label = self.create_binary_label(
            digit=digit_label,
            task_id=task_id
        )

        task_id = torch.tensor(task_id, dtype=torch.long)
        binary_label = torch.tensor(binary_label, dtype=torch.long)
        digit_label = torch.tensor(digit_label, dtype=torch.long)

        return image, task_id, binary_label, digit_label

    def create_binary_label(self, digit: int, task_id: int) -> int:
        if task_id == 0:
            return int(digit % 2 == 0)

        elif task_id == 1:
            return int(digit % 2 == 1)

        elif task_id == 2:
            return int(digit > 5)

        elif task_id == 3:
            return int(digit < 3)

        else:
            raise ValueError(f"Unknown task_id: {task_id}")


# 5. Model Definitions
class ImageEncoder(nn.Module):
    """
    CNN image encoder.

    Input:
        images: [batch_size, 1, 28, 28]

    Output:
        image_embeddings: [batch_size, image_embedding_dim]
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()

        self.embedding_layer = nn.Linear(
            in_features=32 * 7 * 7,
            out_features=embedding_dim
        )

    def forward(self, images):
        x = self.conv_layers(images)
        x = self.flatten(x)
        image_embeddings = self.embedding_layer(x)

        return image_embeddings


class TaskEncoder(nn.Module):
    """
    Task encoder using nn.Embedding.

    Input:
        task_ids: [batch_size]

    Output:
        task_embeddings: [batch_size, task_embedding_dim]
    """

    def __init__(self, num_tasks: int = 4, task_embedding_dim: int = 16):
        super().__init__()

        self.task_embedding = nn.Embedding(
            num_embeddings=num_tasks,
            embedding_dim=task_embedding_dim
        )

    def forward(self, task_ids):
        task_embeddings = self.task_embedding(task_ids)

        return task_embeddings


class ImageOnlyClassifier(nn.Module):
    """
    Baseline model.

    Uses only the image.
    Ignores task_id.
    """

    def __init__(self, image_embedding_dim: int = 128, num_classes: int = 2):
        super().__init__()

        self.image_encoder = ImageEncoder(
            embedding_dim=image_embedding_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(image_embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, images, task_ids=None):
        image_embeddings = self.image_encoder(images)
        logits = self.classifier(image_embeddings)

        return logits


class TaskOnlyClassifier(nn.Module):
    """
    Baseline model.

    Uses only the task_id.
    Ignores the image.
    """

    def __init__(
        self,
        num_tasks: int = 4,
        task_embedding_dim: int = 16,
        num_classes: int = 2
    ):
        super().__init__()

        self.task_encoder = TaskEncoder(
            num_tasks=num_tasks,
            task_embedding_dim=task_embedding_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(task_embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, images, task_ids):
        task_embeddings = self.task_encoder(task_ids)
        logits = self.classifier(task_embeddings)

        return logits


class FusionClassifier(nn.Module):
    """
    Full multimodal fusion model.

    Uses:
        image -> image encoder -> image embedding
        task_id -> task encoder -> task embedding

    Then:
        concatenate image embedding + task embedding
        classify yes/no
    """

    def __init__(
        self,
        image_embedding_dim: int = 128,
        task_embedding_dim: int = 16,
        num_tasks: int = 4,
        num_classes: int = 2
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(
            embedding_dim=image_embedding_dim
        )

        self.task_encoder = TaskEncoder(
            num_tasks=num_tasks,
            task_embedding_dim=task_embedding_dim
        )

        fusion_input_dim = image_embedding_dim + task_embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, images, task_ids):
        image_embeddings = self.image_encoder(images)

        task_embeddings = self.task_encoder(task_ids)

        combined = torch.cat(
            [image_embeddings, task_embeddings],
            dim=1
        )

        logits = self.classifier(combined)

        return logits


# 6. Training Function
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, task_ids, labels, digit_labels in train_loader:
        images = images.to(device)
        task_ids = task_ids.to(device)
        labels = labels.to(device)

        logits = model(images, task_ids)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size

        predictions = torch.argmax(logits, dim=1)
        total_correct += (predictions == labels).sum().item()
        total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


# 7. Evaluation Function
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion,
    device: torch.device
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, task_ids, labels, digit_labels in data_loader:
            images = images.to(device)
            task_ids = task_ids.to(device)
            labels = labels.to(device)

            logits = model(images, task_ids)

            loss = criterion(logits, labels)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size

            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


# 8. Per-Task Accuracy
def evaluate_per_task(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_tasks: int = 4
) -> Dict[int, float]:
    model.eval()

    correct_by_task = {task_id: 0 for task_id in range(num_tasks)}
    total_by_task = {task_id: 0 for task_id in range(num_tasks)}

    with torch.no_grad():
        for images, task_ids, labels, digit_labels in data_loader:
            images = images.to(device)
            task_ids = task_ids.to(device)
            labels = labels.to(device)

            logits = model(images, task_ids)
            predictions = torch.argmax(logits, dim=1)

            for task_id in range(num_tasks):
                mask = task_ids == task_id

                if mask.sum().item() == 0:
                    continue

                task_predictions = predictions[mask]
                task_labels = labels[mask]

                correct = (task_predictions == task_labels).sum().item()
                total = task_labels.size(0)

                correct_by_task[task_id] += correct
                total_by_task[task_id] += total

    accuracy_by_task = {}

    for task_id in range(num_tasks):
        if total_by_task[task_id] > 0:
            accuracy_by_task[task_id] = correct_by_task[task_id] / total_by_task[task_id]
        else:
            accuracy_by_task[task_id] = 0.0

    return accuracy_by_task


# 9. Failure Collection
def collect_failures(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_failures: int = 10
) -> List[dict]:
    model.eval()

    failures = []

    with torch.no_grad():
        for images, task_ids, labels, digit_labels in data_loader:
            images = images.to(device)
            task_ids = task_ids.to(device)
            labels = labels.to(device)
            digit_labels = digit_labels.to(device)

            logits = model(images, task_ids)
            predictions = torch.argmax(logits, dim=1)

            wrong_mask = predictions != labels
            wrong_indices = torch.where(wrong_mask)[0]

            for idx in wrong_indices:
                task_id = task_ids[idx].item()

                failure = {
                    "digit": digit_labels[idx].item(),
                    "task_id": task_id,
                    "task_name": TASK_NAMES[task_id],
                    "prediction": predictions[idx].item(),
                    "true_label": labels[idx].item()
                }

                failures.append(failure)

                if len(failures) >= max_failures:
                    return failures

    return failures


# 10. Experiment Runner
def run_experiment(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    num_epochs: int
) -> dict:
    print("\n" + "=" * 70)
    print(f"Training model: {model_name}")
    print("=" * 70)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        test_loss, test_accuracy = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Test Loss: {test_loss:.4f} "
            f"Test Acc: {test_accuracy:.4f}"
        )

    final_test_loss, final_test_accuracy = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device
    )

    return {
        "model_name": model_name,
        "model": model,
        "test_loss": final_test_loss,
        "test_accuracy": final_test_accuracy
    }


# 11. Results Saving
def save_baseline_results(results: List[dict], output_path: str) -> None:
    lines = []

    lines.append("# Baseline Comparison\n")
    lines.append("| Model | Test Loss | Test Accuracy |")
    lines.append("|---|---:|---:|")

    for result in results:
        lines.append(
            f"| {result['model_name']} | "
            f"{result['test_loss']:.4f} | "
            f"{result['test_accuracy']:.4f} |"
        )

    lines.append("\n## Interpretation\n")
    lines.append(
        "The image-only baseline tests performance without task context. "
        "The task-only baseline tests performance without visual input. "
        "The fusion model tests whether combining both modalities improves performance.\n"
    )

    lines.append(
        "If the fusion model outperforms both baselines, that suggests the task requires "
        "both visual information and task context.\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_failure_analysis(
    failures: List[dict],
    per_task_accuracy: Dict[int, float],
    output_path: str
) -> None:
    lines = []

    lines.append("# Failure Analysis\n")

    lines.append("## Per-Task Accuracy\n")
    lines.append("| Task ID | Task | Accuracy |")
    lines.append("|---:|---|---:|")

    for task_id, accuracy in per_task_accuracy.items():
        lines.append(
            f"| {task_id} | {TASK_NAMES[task_id]} | {accuracy:.4f} |"
        )

    lines.append("\n## Example Failures\n")
    lines.append("| Digit | Task ID | Task | Prediction | True Label |")
    lines.append("|---:|---:|---|---:|---:|")

    for failure in failures:
        lines.append(
            f"| {failure['digit']} | "
            f"{failure['task_id']} | "
            f"{failure['task_name']} | "
            f"{failure['prediction']} | "
            f"{failure['true_label']} |"
        )

    lines.append("\n## Possible Failure Categories\n")
    lines.append("1. Visual recognition error: the CNN may misread the digit.")
    lines.append("2. Task interpretation error: the task embedding may not clearly separate task meanings.")
    lines.append("3. Boundary confusion: greater-than or less-than tasks may produce threshold mistakes.")
    lines.append("4. Dataset prior shortcut: the model may exploit label imbalance.")
    lines.append("5. Undertraining: the model may need more epochs or better hyperparameters.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# 12. Main
def main():
    parser = argparse.ArgumentParser(
        description="Mini multimodal task predictor: MNIST image + task ID -> yes/no"
    )

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default="./data")

    parser.add_argument("--image-embedding-dim", type=int, default=128)
    parser.add_argument("--task-embedding-dim", type=int, default=16)

    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional limit for training samples for quick testing."
    )

    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Optional limit for test samples for quick testing."
    )

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)

    device = get_device()

    print("Using device:", device)

    ensure_dir("results")
    ensure_dir("checkpoints")

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Datasets
    train_dataset = ExpandedMNISTTaskDataset(
        root=args.data_root,
        train=True,
        transform=transform
    )

    test_dataset = ExpandedMNISTTaskDataset(
        root=args.data_root,
        train=False,
        transform=transform
    )

    # Optional smaller dataset for quick runs
    if args.train_limit is not None:
        train_dataset = Subset(
            train_dataset,
            list(range(min(args.train_limit, len(train_dataset))))
        )

    if args.test_limit is not None:
        test_dataset = Subset(
            test_dataset,
            list(range(min(args.test_limit, len(test_dataset))))
        )

    # DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Quick batch shape check
    images, task_ids, labels, digit_labels = next(iter(train_loader))

    print("\nBatch shape check:")
    print("Images:", images.shape)
    print("Task IDs:", task_ids.shape)
    print("Labels:", labels.shape)
    print("Digit labels:", digit_labels.shape)

    # Models
    image_only_model = ImageOnlyClassifier(
        image_embedding_dim=args.image_embedding_dim,
        num_classes=2
    )

    task_only_model = TaskOnlyClassifier(
        num_tasks=4,
        task_embedding_dim=args.task_embedding_dim,
        num_classes=2
    )

    fusion_model = FusionClassifier(
        image_embedding_dim=args.image_embedding_dim,
        task_embedding_dim=args.task_embedding_dim,
        num_tasks=4,
        num_classes=2
    )

    # Run experiments
    results = []

    image_only_result = run_experiment(
        model_name="Image-only baseline",
        model=image_only_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.lr,
        num_epochs=args.epochs
    )

    results.append(image_only_result)

    task_only_result = run_experiment(
        model_name="Task-only baseline",
        model=task_only_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.lr,
        num_epochs=args.epochs
    )

    results.append(task_only_result)

    fusion_result = run_experiment(
        model_name="Image + task fusion model",
        model=fusion_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.lr,
        num_epochs=args.epochs
    )

    results.append(fusion_result)

    trained_fusion_model = fusion_result["model"]

    # Final comparison
    print("\n" + "=" * 70)
    print("Final Comparison")
    print("=" * 70)

    for result in results:
        print(
            f"{result['model_name']}: "
            f"Test Loss = {result['test_loss']:.4f}, "
            f"Test Accuracy = {result['test_accuracy']:.4f}"
        )

    # Per-task accuracy for fusion model
    per_task_accuracy = evaluate_per_task(
        model=trained_fusion_model,
        data_loader=test_loader,
        device=device,
        num_tasks=4
    )

    print("\nFusion Model Per-Task Accuracy:")
    for task_id, accuracy in per_task_accuracy.items():
        print(f"Task {task_id} ({TASK_NAMES[task_id]}): {accuracy:.4f}")

    # Failure examples
    failures = collect_failures(
        model=trained_fusion_model,
        data_loader=test_loader,
        device=device,
        max_failures=10
    )

    print("\nExample Fusion Model Failures:")
    if len(failures) == 0:
        print("No failures found in inspected test batches.")
    else:
        for i, failure in enumerate(failures):
            print(
                f"Failure {i + 1}: "
                f"digit={failure['digit']}, "
                f"task={failure['task_name']}, "
                f"prediction={failure['prediction']}, "
                f"true_label={failure['true_label']}"
            )

    # Save results
    save_baseline_results(
        results=results,
        output_path="results/baseline_comparison.md"
    )

    save_failure_analysis(
        failures=failures,
        per_task_accuracy=per_task_accuracy,
        output_path="results/failure_analysis.md"
    )

    # Save fusion model checkpoint
    torch.save(
        trained_fusion_model.state_dict(),
        "checkpoints/fusion_model.pt"
    )

    print("\nSaved files:")
    print("- results/baseline_comparison.md")
    print("- results/failure_analysis.md")
    print("- checkpoints/fusion_model.pt")


if __name__ == "__main__":
    main()
