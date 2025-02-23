
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import numpy as np
from typing import List, Callable, Dict, Any
import time
import os
class Callback:
    def on_train_begin(self, logs=None): pass
    def on_train_end(self, logs=None): pass
    def on_epoch_begin(self, epoch, logs=None): pass
    def on_epoch_end(self, epoch, logs=None): pass
    def on_batch_begin(self, batch, logs=None): pass
    def on_batch_end(self, batch, logs=None): pass
    def on_train_batch_begin(self, batch, logs=None): pass
    def on_train_batch_end(self, batch, logs=None): pass
    def on_val_batch_begin(self, batch, logs=None): pass
    def on_val_batch_end(self, batch, logs=None): pass

class MetricCallback(Callback):
    def __init__(self):
        self.metrics = {}

    def on_epoch_begin(self, epoch, logs=None):
        for key in self.metrics:
            self.metrics[key] = []

    def on_batch_end(self, batch, logs=None):
        if logs:
            for key, value in logs.items():
                if key in self.metrics:
                    self.metrics[key].append(value)

class Learner:

    def __init__(self, model, train_loader, val_loader=None, test_loader=None, optimizer=None, loss_fn=None, device=None, callbacks=None):
        self.device = device or (torch.cuda.is_available() and torch.device('cuda')) or torch.device('cpu')

        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.optimizer = optimizer or optim.Adam(self.model.parameters())
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

        self.callbacks = callbacks or []
        self.metrics_callback = MetricCallback()
        self.callbacks.append(self.metrics_callback)

        self.current_epoch = 0
        self.history = {}

        self.scaler = GradScaler('cuda') if torch.cuda.is_available() else None

        for callback in self.callbacks:
            callback.learner = self

    def _run_callbacks(self, event, *args, **kwargs):
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method:
                method(*args, **kwargs)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            if inputs is None or targets is None:
                continue
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

            self._run_callbacks('on_train_batch_begin', batch_idx)

            if self.scaler:
                with autocast('cuda'):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            total_correct += (predicted == targets).sum().item()
            total_loss += loss.item()

            batch_logs = {
                'batch': batch_idx,
                'loss': loss.item(),
                'accuracy': (predicted == targets).float().mean().item()
            }
            self._run_callbacks('on_train_batch_end', batch_idx, logs=batch_logs)

        epoch_loss = total_loss / len(self.train_loader)
        epoch_accuracy = (total_correct / total_samples) * 100

        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy
        }

    def validate(self, loader=None):
        loader = loader or self.val_loader
        if not loader:  
            return {}

        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):  
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self._run_callbacks('on_val_batch_begin', batch_idx)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                _, predicted = torch.max(outputs.data, 1)
                total_samples += targets.size(0)
                total_correct += (predicted == targets).sum().item()
                total_loss += loss.item()

                batch_logs = {
                    'batch': batch_idx,
                    'val_loss': loss.item(),
                    'val_accuracy': (predicted == targets).float().mean().item()
                }
                self._run_callbacks('on_val_batch_end', batch_idx, logs=batch_logs)

        val_loss = total_loss / len(loader)  
        val_accuracy = (total_correct / total_samples) * 100

        return {
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }

    def fit(self, epochs):
        self._run_callbacks('on_train_begin')

        for epoch in range(epochs):
            self.current_epoch = epoch
            self._run_callbacks('on_epoch_begin', epoch)

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            epoch_metrics = {
                'epoch': epoch,
                **train_metrics,
                **val_metrics
            }

            for key, value in epoch_metrics.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)

            self._run_callbacks('on_epoch_end', epoch, logs=epoch_metrics)

        self._run_callbacks('on_train_end')
        return self.history


class PlotCallback(Callback):
    def __init__(self, plot_every=1):
        self.plot_every = plot_every
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            self.train_losses.append(logs.get('loss', 0))
            self.val_losses.append(logs.get('val_loss', 0))
            self.train_accuracies.append(logs.get('accuracy', 0))
            self.val_accuracies.append(logs.get('val_accuracy', 0))
            
            if (epoch + 1) % self.plot_every == 0:
                self.plot_metrics()
    
    def plot_metrics(self):
        import matplotlib.pyplot as plt
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


class PrintCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch}:")
        print(f"  Training Loss: {logs.get('loss', 'NA')}")
        print(f"  Training Accuracy: {logs.get('accuracy', 'NA')}")
        print(f"  Validation Loss: {logs.get('val_loss', 'NA')}")
        print(f"  Validation Accuracy: {logs.get('val_accuracy', 'NA')}")

class TestCallback(Callback):
    def on_train_end(self, logs=None):
        if self.learner.test_loader:
            test_metrics = self.learner.validate(loader=self.learner.test_loader)
            print(f"\nTest Metrics:")
            print(f"  Test Accuracy: {test_metrics['val_accuracy']:.4f}")
