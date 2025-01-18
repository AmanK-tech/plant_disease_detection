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
    def __init__(self): self.reset()
    def reset(self): self.vals, self.ns = [], []
    
    def add(self, inp, targ=None, n=1):
        self.last = self.calc(inp, targ)
        self.vals.append(self.last)
        self.ns.append(n)
    
    @property
    def value(self):
        ns = tensor(self.ns)
        return ((tensor(self.vals) * ns).sum() / ns.sum()).item()
    
    def calc(self, inp, targ): return (inp == targ).float().mean()

class MetricsCB(Callback):
    def __init__(self, *ms, **metrics):
        for m in ms: metrics[m.__class__.__name__] = m
        self.metrics = metrics
        self.all_metrics = copy(metrics)
        self.all_metrics['loss'] = self.loss = Mean()

    def _log(self, d): print(d)
    def before_fit(self, learn): learn.metrics = self
    def before_epoch(self, learn): [m.reset() for m in self.all_metrics.values()]
    
    def after_epoch(self, learn):
        log = {k: f'{v.compute():.3f}' for k,v in self.all_metrics.items()}
        log['epoch'] = learn.epoch
        log['train'] = 'train' if learn.training else 'eval'
        self._log(log)
    
    def after_batch(self, learn):
        x,y,*_ = to_cpu(learn.batch)
        for m in self.metrics.values():
            m.update(to_cpu(learn.preds), y)
        self.loss.update(to_cpu(learn.loss), weight=len(x))

default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_device(obj, device):
    if isinstance(obj, (list,tuple)): return [to_device(o,device) for o in obj]
    if isinstance(obj, dict): return {k: to_device(v,device) for k,v in obj.items()}
    return obj.to(device) if hasattr(obj,'to') else obj

class DeviceCB(Callback):
    def __init__(self, device=default_device): self.device = device
    def before_fit(self, learn): learn.model.to(self.device)
    def before_batch(self, learn): learn.batch = to_device(learn.batch, device=self.device)

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
    def __init__(self, model, dls=None, loss_func=F.mse_loss, lr=0.1, cbs=None, opt_func=optim.Adam):
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
        self.cbs = fc.L([] if cbs is None else cbs)
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

    def callback(self, method_nm): run_cbs(self.cbs, method_nm, self)
    
    @property
    def training(self): return self.model.training

class TrainCB(Callback):
    def __init__(self, n_inp=1): self.n_inp = n_inp
    def predict(self, learn): learn.preds = learn.model(*learn.batch[:self.n_inp])
    def get_loss(self, learn): learn.loss = learn.loss_func(learn.preds, *learn.batch[self.n_inp:])
    def backward(self, learn): learn.loss.backward()
    def step(self, learn): learn.opt.step()
    def zero_grad(self, learn): learn.opt.zero_grad()

class ProgressCB(Callback):
    order = MetricsCB.order + 1
    
    def __init__(self, plot=False): 
        self.plot = plot
    
    def before_fit(self, learn):
        
        learn.epochs = list(range(learn.n_epochs))
        self.mbar = master_bar(learn.epochs)
        self.first = True
        if hasattr(learn, 'metrics'): 
            learn.metrics._log = self._log
        self.losses, self.val_losses = [], []
    
    def _log(self, d):
        if self.first:
            self.mbar.write(list(d), table=True)
            self.first = False
        self.mbar.write(list(d.values()), table=True)
    
    def before_epoch(self, learn):
        if not isinstance(learn.dl, LenDataLoader):
            learn.dl = LenDataLoader(learn.dl)
        batches = list(range(len(learn.dl)))
        learn.dl = progress_bar(batches, leave=False, parent=self.mbar)
        self.iter = iter(learn.dl.iterable)
    
    def after_batch(self, learn):
        if hasattr(learn, 'loss'):
            
            learn.dl.comment = f'{float(learn.loss):.3f}'
            if self.plot and learn.training:
                self.losses.append(float(learn.loss))
    
    def after_epoch(self, learn):
        if not learn.training and self.plot and hasattr(learn, 'metrics'):
            val_loss = learn.metrics.all_metrics['loss'].compute()
            self.val_losses.append(float(val_loss))
            if hasattr(self.mbar, 'update_graph'):
                self.mbar.update_graph([[i, l] for i, l in enumerate(self.losses)],
                                     [[i, l] for i, l in enumerate(self.val_losses)])

class LRFinderCB(Callback):
    def __init__(self, gamma=1.3, max_mult=3): 
        self.gamma,self.max_mult = gamma,max_mult
    
    def before_fit(self, learn):
        self.sched = ExponentialLR(learn.opt, self.gamma)
        self.lrs,self.losses = [],[]
        self.min = math.inf
    
    def after_batch(self, learn):
        if not learn.training: raise CancelEpochException()
        self.lrs.append(learn.opt.param_groups[0]['lr'])
        loss = to_cpu(learn.loss).item()
        self.losses.append(loss)
        if loss < self.min: self.min = loss
        if math.isnan(loss) or loss > self.min * self.max_mult:
            raise CancelFitException()
        self.sched.step()
    
    def cleanup_fit(self, learn):
        plt.plot(self.lrs, self.losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.show()

class CompletionCB(Callback):
    def before_fit(self, learn): self.count = 0
    def after_batch(self, learn): self.count += 1
    def after_fit(self, learn): print(f'Completed {self.count} batches')
