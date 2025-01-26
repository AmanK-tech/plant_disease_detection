
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
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
    def __init__(self, model, train_loader, val_loader=None, test_loader=None, optimizer=None, loss_fn=None, device=None,callbacks=None):
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
        
        self.scaler = GradScaler()

    def _run_callbacks(self, event, *args, **kwargs):
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method:
                method(*args, **kwargs)

    def train_epoch(self):
        # self.model.train()
        # total_loss = 0
        # total_correct = 0
        # total_samples = 0

        # self._run_callbacks('on_epoch_begin', self.current_epoch)

        # for batch_idx, (inputs, targets) in enumerate(self.train_loader):
        #     if inputs is None or targets is None:
        #       continue
        #     inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        print(f"Total batches in train_loader: {len(self.train_loader)}")

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            print(f"Batch {batch_idx}:")
            if inputs is None or targets is None:
                print(f"  Skipping batch {batch_idx} - inputs or targets are None")
                continue
            
            print(f"  Inputs shape: {inputs.shape}")
            print(f"  Targets shape: {targets.shape}")
            
            inputs, targets = inputs.to(self.device), targets.to(self.device)
        
            
            self._run_callbacks('on_train_batch_begin', batch_idx)
            
            with autocast():
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

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
        epoch_accuracy = total_correct / total_samples

        epoch_logs = {
            'epoch': self.current_epoch,
            'loss': epoch_loss,
            'accuracy': epoch_accuracy
        }
        self._run_callbacks('on_epoch_end', self.current_epoch, logs=epoch_logs)

        return epoch_logs

    def validate(self):
        if not self.val_loader:
            return None

        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
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
                    'loss': loss.item(),
                    'accuracy': (predicted == targets).float().mean().item()
                }
                self._run_callbacks('on_val_batch_end', batch_idx, logs=batch_logs)

        val_loss = total_loss / len(self.val_loader)
        val_accuracy = total_correct / total_samples

        return {
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }

    def fit(self, epochs):
        self._run_callbacks('on_train_begin')
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch()
            
            val_metrics = self.validate() if self.val_loader else {}
            
            epoch_metrics = {**train_metrics, **val_metrics}
            
            for key, value in epoch_metrics.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)

        self._run_callbacks('on_train_end')
        return self.history

class PrintCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch}: {logs}")



    
