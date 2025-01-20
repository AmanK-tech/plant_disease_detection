import torch
from torch.optim.lr_scheduler import ExponentialLR
from torcheval.metrics import MulticlassAccuracy, Mean
from torch import optim
import torch.nn.functional as F
from torch import tensor
from collections.abc import Mapping
from operator import attrgetter
from functools import partial
from copy import copy
import fastcore.foundation as fc
from fastprogress.fastprogress import master_bar, progress_bar
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader

# Exception classes for control flow
class CancelFitException(Exception): pass
class CancelBatchException(Exception): pass
class CancelEpochException(Exception): pass

class LenDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        
        if hasattr(dataloader, 'dataset'):
            self.dataset = dataloader.dataset
            self.dataset_len = len(self.dataset)
        else:
            self.dataset = None
            self.dataset_len = len(dataloader) * dataloader.batch_size
        
        self.batch_size = dataloader.batch_size
        self._length = (self.dataset_len + self.batch_size - 1) // self.batch_size
        
        self.batch_sampler = getattr(dataloader, 'batch_sampler', None)
        self.sampler = getattr(dataloader, 'sampler', None)
        self.collate_fn = getattr(dataloader, 'collate_fn', None)
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return self._length

class Callback:
    order = 0

def run_cbs(cbs, method_nm, learn=None):
    for cb in sorted(cbs, key=attrgetter('order')):
        method = getattr(cb, method_nm, None)
        if method is not None: method(learn)

def to_cpu(x):
    if isinstance(x, Mapping): return {k: to_cpu(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return type(x)(to_cpu(o) for o in x)
    return x.detach().cpu() if isinstance(x, torch.Tensor) else x

class Metric:
    def __init__(self): 
        self.reset()
    
    def reset(self): 
        self.vals, self.ns = [], []
        self.total = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.total += val * n
        self.count += n
        self.vals.append(val)
        self.ns.append(n)
    
    def compute(self):
        return self.total / self.count if self.count > 0 else 0.0

class AccuracyMetric(Metric):
    def update(self, preds, targets):
        correct = (preds.argmax(dim=1) == targets).float().sum().item()
        total = targets.numel()
        super().update(correct / total, total)

class MetricsCB(Callback):
    def __init__(self, metrics=None):
        self.metrics = {} if metrics is None else metrics
        self.all_metrics = copy(self.metrics)
        self.all_metrics['loss'] = Mean()
        self.all_metrics['accuracy'] = AccuracyMetric()
    
    def _log(self, d):
        phase = 'Training' if d['train'] == 'train' else 'Validation'
        metrics_str = ', '.join(f"{k.title()}: {v:.4f}" for k, v in d.items() 
                              if k not in ['epoch', 'train'])
        print(f"{phase:<10} - {metrics_str}")
    
    def before_fit(self, learn): 
        learn.metrics = self
    
    def before_epoch(self, learn): 
        [m.reset() for m in self.all_metrics.values()]
    
    def after_batch(self, learn):
        if hasattr(learn, 'loss'):
            self.all_metrics['loss'].update(learn.loss.item())
        if hasattr(learn, 'preds'):
            self.all_metrics['accuracy'].update(learn.preds, learn.batch[1])
    
    def after_epoch(self, learn):
        log = {k: v.compute() for k,v in self.all_metrics.items()}
        log['epoch'] = learn.epoch
        log['train'] = 'train' if learn.training else 'eval'
        self._log(log)

class DeviceCB(Callback):
    order = 0
    
    def __init__(self, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        
        if self.device.type == 'cuda':
            print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
        else:
            print("\nNo GPU available, using CPU")
    
    def before_fit(self, learn):
        learn.model.to(self.device)
        if hasattr(learn, 'opt'):
            for state in learn.opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
    
    def before_batch(self, learn):
        learn.batch = to_device(learn.batch, self.device)

class with_cbs:
    def __init__(self, nm): self.nm = nm
    def __call__(self, f):
        def _f(o, *args, **kwargs):
            try:
                o.callback(f'before_{self.nm}')
                f(o, *args, **kwargs)
                o.callback(f'after_{self.nm}')
            except globals()[f'Cancel{self.nm.title()}Exception']: pass
            finally: o.callback(f'cleanup_{self.nm}')
        return _f

class Learner:
    def __init__(self, model, dls=None, loss_func=F.cross_entropy, lr=0.1, cbs=None, opt_func=optim.Adam):
        self.model = model
        
        if dls is None:
            self.dls = {'train': None, 'valid': None}
        elif isinstance(dls, dict):
            self.dls = {k: LenDataLoader(v) if isinstance(v, DataLoader) else v
                       for k, v in dls.items()}
        elif isinstance(dls, (list, tuple)):
            train_dl = LenDataLoader(dls[0]) if isinstance(dls[0], DataLoader) else dls[0]
            valid_dl = (LenDataLoader(dls[1]) if isinstance(dls[1], DataLoader) else dls[1]) if len(dls) > 1 else None
            self.dls = {'train': train_dl, 'valid': valid_dl}
        else:
            self.dls = {'train': LenDataLoader(dls) if isinstance(dls, DataLoader) else dls,
                       'valid': None}
        
        self.loss_func = loss_func
        self.lr = lr
        self.cbs = fc.L([MetricsCB(), DeviceCB()])  # Default callbacks
        if cbs is not None:
            self.cbs.extend(cbs)
        self.opt_func = opt_func
    
    @with_cbs('batch')
    def _one_batch(self):
        self.predict()
        self.callback('after_predict')
        self.get_loss()
        self.callback('after_loss')
        if self.training:
            self.backward()
            self.callback('after_backward')
            self.step()
            self.callback('after_step')
            self.zero_grad()
    
    @with_cbs('epoch')
    def _one_epoch(self):
        for self.iter,self.batch in enumerate(self.dl):
            self._one_batch()
    
    def one_epoch(self, training):
        self.model.train(training)
        self.dl = self.dls['train'] if training else self.dls['valid']
        if self.dl is not None:
            self._one_epoch()
    
    @with_cbs('fit')
    def _fit(self, train, valid):
        for self.epoch in range(self.n_epochs):
            if train: self.one_epoch(True)
            if valid and self.dls['valid'] is not None:
                with torch.no_grad(): self.one_epoch(False)
    
    def fit(self, n_epochs=1, train=True, valid=True, cbs=None, lr=None):
        cbs = fc.L([] if cbs is None else cbs)
        for cb in cbs: self.cbs.append(cb)
        try:
            self.n_epochs = n_epochs
            self.epochs = range(n_epochs)
            if lr is None: lr = self.lr
            self.opt = self.opt_func(self.model.parameters(), lr)
            self._fit(train, valid)
        finally:
            for cb in cbs: self.cbs.remove(cb)
    
    def __getattr__(self, name):
        if name in ('predict','get_loss','backward','step','zero_grad'):
            return partial(self.callback, name)
        raise AttributeError(name)
    
    def callback(self, method_nm): 
        run_cbs(self.cbs, method_nm, self)
    
    @property
    def training(self): 
        return self.model.training

class TrainCB(Callback):
    def __init__(self, n_inp=1): 
        self.n_inp = n_inp
    
    def predict(self, learn): 
        learn.preds = learn.model(*learn.batch[:self.n_inp])
    
    def get_loss(self, learn): 
        learn.loss = learn.loss_func(learn.preds, learn.batch[self.n_inp])
    
    def backward(self, learn): 
        learn.loss.backward()
    
    def step(self, learn): 
        learn.opt.step()
    
    def zero_grad(self, learn): 
        learn.opt.zero_grad()

class ProgressCallback(Callback):
    order = MetricsCB.order + 1
    
    def __init__(self, plot=True):
        self.plot = plot
        self.train_losses = []
        self.valid_losses = []
        self.train_accs = []
        self.valid_accs = []
    
    def before_fit(self, learn):
        print(f"\nStarting training for {learn.n_epochs} epochs\n")
    
    def after_epoch(self, learn):
        if hasattr(learn, 'metrics'):
            metrics = learn.metrics.all_metrics
            loss = metrics['loss'].compute()
            acc = metrics['accuracy'].compute()
            
            if learn.training:
                self.train_losses.append(loss)
                self.train_accs.append(acc)
            else:
                self.valid_losses.append(loss)
                self.valid_accs.append(acc)
        
        if self.plot and learn.epoch == learn.n_epochs-1:
            self._plot_metrics()
    
    def _plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        if self.valid_losses:
            plt.plot(epochs, self.valid_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accs, 'b-', label='Training Accuracy')
        if self.valid_accs:
            plt.plot(epochs, self.valid_accs, 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
