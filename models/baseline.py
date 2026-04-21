"""
Module 5: Letter-Level Baseline Classifier

Simple CNN classifier on individual clean letter recordings.
Ablation condition ①: just classify one letter at a time.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Tuple, List
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LetterClassifier(nn.Module):
    """
    CNN classifier for Braille letter recognition.
    
    Input: single letter recording (T, 12)
    Outputs: class logits for 27 characters (a-z + space)
    
    Architecture:
        Conv1D(12 → 32, k=5) + ReLU + MaxPool
        Conv1D(32 → 64, k=3) + ReLU + GlobalAvgPool
        Linear(64 → 27) + Softmax
    """
    
    def __init__(self, num_classes: int = 27, fixed_length: int = 150):
        """
        Initialize classifier.
        
        Args:
            num_classes: Number of output classes (27 for a-z + space)
            fixed_length: Fixed input length (T=150)
        """
        super().__init__()
        self.num_classes = num_classes
        self.fixed_length = fixed_length
        
        self.conv1 = nn.Conv1d(12, 32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input (B, 12, T) or (1, 12, T)
        
        Returns:
            Logits (B, 27)
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        x = self.global_avgpool(x)
        x = x.squeeze(-1)  # (B, 64)
        
        x = self.fc(x)  # (B, 27)
        
        return x


class LetterClassifierTrainer:
    """Trainer for letter classifier."""
    
    def __init__(self,
                 model: LetterClassifier,
                 device: str = 'cpu',
                 learning_rate: float = 1e-3):
        """
        Initialize trainer.
        
        Args:
            model: Classifier model
            device: 'cpu' or 'cuda'
            learning_rate: Adam learning rate
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def prepare_batch(self, signals: list, labels: list,
                     fixed_length: int = 150) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch of variable-length signals.
        
        Args:
            signals: List of (T, 12) arrays
            labels: List of class indices
            fixed_length: Target length (pad/truncate to this)
        
        Returns:
            Tuple of (signal_batch, label_batch)
        """
        batch_signals = []
        
        for signal in signals:
            T = signal.shape[0]
            
            if T >= fixed_length:
                # Truncate to fixed_length
                signal_proc = signal[:fixed_length, :]
            else:
                # Pad to fixed_length
                pad_len = fixed_length - T
                signal_proc = np.pad(signal, ((0, pad_len), (0, 0)),
                                    mode='constant', constant_values=0)
            
            batch_signals.append(signal_proc)
        
        # Convert to tensor: (B, T, 12) -> (B, 12, T)
        signal_batch = torch.from_numpy(np.array(batch_signals)).transpose(1, 2).float().to(self.device)
        label_batch = torch.from_numpy(np.array(labels)).long().to(self.device)
        
        return signal_batch, label_batch
    
    def train_epoch(self, train_data, batch_size: int = 32) -> float:
        """Train one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle
        indices = np.random.permutation(len(train_data))
        
        for batch_start in range(0, len(train_data), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch_signals = [train_data[i][0] for i in batch_indices]
            batch_labels = [train_data[i][1] for i in batch_indices]
            
            signal_batch, label_batch = self.prepare_batch(batch_signals, batch_labels)
            
            self.optimizer.zero_grad()
            output = self.model(signal_batch)
            loss = self.criterion(output, label_batch)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def evaluate(self, data, batch_size: int = 32) -> Tuple[float, float]:
        """
        Evaluate on a dataset.
        
        Args:
            data: List of (signal, label) tuples
            batch_size: Batch size
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_start in range(0, len(data), batch_size):
                batch_signals = [data[i][0] for i in range(batch_start, min(batch_start + batch_size, len(data)))]
                batch_labels = [data[i][1] for i in range(batch_start, min(batch_start + batch_size, len(data)))]
                
                signal_batch, label_batch = self.prepare_batch(batch_signals, batch_labels)
                
                output = self.model(signal_batch)
                loss = self.criterion(output, label_batch)
                
                # Accuracy
                preds = output.argmax(dim=1)
                correct = (preds == label_batch).sum().item()
                
                total_loss += loss.item()
                total_correct += correct
                total_samples += len(batch_labels)
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return avg_loss, accuracy
    
    def fit(self, train_data, val_data,
            epochs: int = 50, batch_size: int = 32,
            patience: int = 10, checkpoint_dir: str = "checkpoints"):
        """
        Train classifier with early stopping.
        
        Args:
            train_data: List of (signal, label) tuples
            val_data: List of (signal, label) tuples
            epochs: Maximum epochs
            batch_size: Batch size
            patience: Early stopping patience
            checkpoint_dir: Checkpoint directory
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("TRAINING LETTER CLASSIFIER (BASELINE)")
        logger.info("=" * 60)
        logger.info(f"Train samples: {len(train_data)}")
        logger.info(f"Val samples: {len(val_data)}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, Patience: {patience}")
        logger.info("=" * 60)
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_data, batch_size)
            val_loss, val_acc = self.evaluate(val_data, batch_size)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            log_msg = (
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                
                checkpoint_path = os.path.join(checkpoint_dir, 'letter_baseline.pt')
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info(log_msg + " ✓ (saved)")
            else:
                self.patience_counter += 1
                logger.info(log_msg)
                
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            self.scheduler.step(val_loss)
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best val accuracy: {self.best_val_acc:.4f}")
        logger.info("=" * 60)
    
    def test(self, test_data, batch_size: int = 32) -> Tuple[float, dict]:
        """
        Test on test set and return per-character accuracy.
        
        Args:
            test_data: List of (signal, label) tuples
            batch_size: Batch size
        
        Returns:
            Tuple of (overall_accuracy, per_char_accuracy_dict)
        """
        loss, overall_acc = self.evaluate(test_data, batch_size)
        
        # Per-character accuracy
        per_char_correct = {}
        per_char_total = {}
        
        self.model.eval()
        with torch.no_grad():
            for signal, label in test_data:
                signal_tensor = torch.from_numpy(signal).float().to(self.device).unsqueeze(0).transpose(1, 2)
                output = self.model(signal_tensor)
                pred = output.argmax(dim=1).item()
                
                char = 'abcdefghijklmnopqrstuvwxyz '[label]
                
                if char not in per_char_total:
                    per_char_total[char] = 0
                    per_char_correct[char] = 0
                
                per_char_total[char] += 1
                if pred == label:
                    per_char_correct[char] += 1
        
        per_char_acc = {
            char: per_char_correct[char] / per_char_total[char]
            for char in per_char_total
        }
        
        return overall_acc, per_char_acc
