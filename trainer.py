# import time
# from datetime import timedelta

# import torch
# from torch.nn import Module
# from torch.optim import Optimizer
# from torch.utils.data import DataLoader
# from tqdm.auto import tqdm
# from transformers import PreTrainedTokenizer

# from lettucedetect.models.evaluator import evaluate_model, print_metrics


# class Trainer:
#     """Token classification trainer with epoch-based training and validation.

#     Trains a model using AdamW, evaluates on a test set after each epoch,
#     and saves the best checkpoint based on hallucinated-class F1.
#     """

#     def __init__(
#         self,
#         model: Module,
#         tokenizer: PreTrainedTokenizer,
#         train_loader: DataLoader,
#         test_loader: DataLoader,
#         epochs: int = 6,
#         learning_rate: float = 1e-5,
#         save_path: str = "best_model",
#         device: torch.device | None = None,
#     ):
#         """Initialize the trainer.

#         :param model: The model to train
#         :param tokenizer: Tokenizer for the model
#         :param train_loader: DataLoader for training data
#         :param test_loader: DataLoader for test data
#         :param epochs: Number of training epochs
#         :param learning_rate: Learning rate for optimization
#         :param save_path: Path to save the best model
#         :param device: Device to train on (defaults to cuda if available)
#         """
#         self.model = model
#         self.tokenizer = tokenizer
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.epochs = epochs
#         self.learning_rate = learning_rate
#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.save_path = save_path

#         self.optimizer: Optimizer = torch.optim.AdamW(
#             self.model.parameters(), lr=self.learning_rate
#         )
#         self.model.to(self.device)

#     def train(self) -> float:
#         """Train the model.

#         Returns:
#             Best F1 score achieved during training

#         """
#         best_f1: float = 0
#         start_time = time.time()

#         print(f"\nStarting training on {self.device}")
#         print(
#             f"Training samples: {len(self.train_loader.dataset)}, "
#             f"Test samples: {len(self.test_loader.dataset)}\n"
#         )

#         for epoch in range(self.epochs):
#             epoch_start = time.time()
#             print(f"\nEpoch {epoch + 1}/{self.epochs}")

#             self.model.train()
#             total_loss = 0
#             num_batches = 0

#             progress_bar = tqdm(self.train_loader, desc="Training", leave=True)

#             for batch in progress_bar:
#                 self.optimizer.zero_grad()
#                 outputs = self.model(
#                     batch["input_ids"].to(self.device),
#                     attention_mask=batch["attention_mask"].to(self.device),
#                     labels=batch["labels"].to(self.device),
#                 )
#                 loss = outputs.loss
#                 loss.backward()
#                 self.optimizer.step()

#                 total_loss += loss.item()
#                 num_batches += 1

#                 progress_bar.set_postfix(
#                     {
#                         "loss": f"{loss.item():.4f}",
#                         "avg_loss": f"{total_loss / num_batches:.4f}",
#                     }
#                 )

#             avg_loss = total_loss / num_batches
#             epoch_time = time.time() - epoch_start
#             print(
#                 f"Epoch {epoch + 1} completed in {timedelta(seconds=int(epoch_time))}. Average loss: {avg_loss:.4f}"
#             )

#             print("\nEvaluating...")
#             metrics = evaluate_model(self.model, self.test_loader, self.device)
#             print_metrics(metrics)

#             if metrics["hallucinated"]["f1"] > best_f1:
#                 best_f1 = metrics["hallucinated"]["f1"]
#                 self.model.save_pretrained(self.save_path)
#                 self.tokenizer.save_pretrained(self.save_path)
#                 print(f"\n🎯 New best F1: {best_f1:.4f}, model saved at '{self.save_path}'!")

#             print("-" * 50)

#         total_time = time.time() - start_time
#         print(f"\nTraining completed in {timedelta(seconds=int(total_time))}")
#         print(f"Best F1 score: {best_f1:.4f}")

#         return best_f1


import time
from datetime import timedelta
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from lettucedetect.models.evaluator import evaluate_model, print_metrics

class FocalLoss(torch.nn.Module):
    """Focal Loss with ignore_index support."""
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = 0.25, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (batch, seq_len, num_labels)
        # targets: (batch, seq_len)
        mask = targets != self.ignore_index
        # Replace ignored indices with 0 temporarily (they will be masked out)
        safe_targets = targets.clone()
        safe_targets[~mask] = 0

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Gather log probabilities and probabilities for the target class
        log_probs_target = log_probs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
        probs_target = probs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)

        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - probs_target) ** self.gamma

        # Apply alpha balancing (for binary classification)
        if self.alpha is not None:
            alpha_t = torch.where(safe_targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_probs_target
        # Mask out ignored positions
        loss = loss[mask]
        return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=logits.device)

class Trainer:
    """Token classification trainer with support for CE, Focal, or mixed loss."""

    def __init__(
        self,
        model: Module,
        tokenizer: PreTrainedTokenizer,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 6,
        learning_rate: float = 1e-5,
        save_path: str = "best_model",
        device: Optional[torch.device] = None,
        # Loss configuration
        loss_type: Literal["ce", "focal", "mixed"] = "mixed",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        ce_weight: float = 0.5,
        focal_weight: float = 0.5,
    ):
        """Initialize the trainer.

        :param model: The model to train
        :param tokenizer: Tokenizer for the model
        :param train_loader: DataLoader for training data
        :param test_loader: DataLoader for test data
        :param epochs: Number of training epochs
        :param learning_rate: Learning rate for optimization
        :param save_path: Path to save the best model
        :param device: Device to train on (defaults to cuda if available)
        :param loss_type: 'ce' (standard cross entropy), 'focal', or 'mixed'
        :param focal_gamma: Focusing parameter for Focal Loss (default 2.0)
        :param focal_alpha: Balancing parameter for Focal Loss (default 0.25)
        :param ce_weight: Weight for CE term when loss_type='mixed'
        :param focal_weight: Weight for Focal term when loss_type='mixed'
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path

        self.optimizer: Optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )
        self.model.to(self.device)

        # Loss setup
        self.loss_type = loss_type
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, ignore_index=-100)
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight

        # Print loss configuration
        print(f"Using loss type: {loss_type}")
        if loss_type == "focal":
            print(f"  Focal Loss: gamma={focal_gamma}, alpha={focal_alpha}")
        elif loss_type == "mixed":
            print(f"  Mixed Loss: CE weight={ce_weight}, Focal weight={focal_weight}")

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss based on the configured loss type."""
        # Reshape for token-level loss: (batch*seq_len, num_labels) vs (batch*seq_len)
        num_labels = logits.shape[-1]
        logits_flat = logits.view(-1, num_labels)
        labels_flat = labels.view(-1)

        if self.loss_type == "ce":
            return F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
        elif self.loss_type == "focal":
            return self.focal_loss(logits, labels)  # expects original shape
        else:  # mixed
            ce_loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
            foc_loss = self.focal_loss(logits, labels)
            return self.ce_weight * ce_loss + self.focal_weight * foc_loss

    def train(self) -> float:
        """Train the model.

        Returns:
            Best F1 score achieved during training
        """
        best_f1: float = 0
        start_time = time.time()

        print(f"\nStarting training on {self.device}")
        print(
            f"Training samples: {len(self.train_loader.dataset)}, "
            f"Test samples: {len(self.test_loader.dataset)}\n"
        )

        for epoch in range(self.epochs):
            epoch_start = time.time()
            print(f"\nEpoch {epoch + 1}/{self.epochs}")

            self.model.train()
            total_loss = 0
            num_batches = 0

            progress_bar = tqdm(self.train_loader, desc="Training", leave=True)

            for batch in progress_bar:
                self.optimizer.zero_grad()

                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass without passing labels to model (to get raw logits)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits  # (batch, seq_len, num_labels)

                # Compute custom loss
                loss = self._compute_loss(logits, labels)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "avg_loss": f"{total_loss / num_batches:.4f}",
                    }
                )

            avg_loss = total_loss / num_batches
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch + 1} completed in {timedelta(seconds=int(epoch_time))}. Average loss: {avg_loss:.4f}"
            )

            print("\nEvaluating...")
            metrics = evaluate_model(self.model, self.test_loader, self.device)
            print_metrics(metrics)

            if metrics["hallucinated"]["f1"] > best_f1:
                best_f1 = metrics["hallucinated"]["f1"]
                self.model.save_pretrained(self.save_path)
                self.tokenizer.save_pretrained(self.save_path)
                print(f"\n🎯 New best F1: {best_f1:.4f}, model saved at '{self.save_path}'!")

            print("-" * 50)

        total_time = time.time() - start_time
        print(f"\nTraining completed in {timedelta(seconds=int(total_time))}")
        print(f"Best F1 score: {best_f1:.4f}")

        return best_f1