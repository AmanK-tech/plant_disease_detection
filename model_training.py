
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

class CancelFitException(Exception): pass
class CancelBatchException(Exception): pass
class CancelEpochException(Exception): pass

class Callback:
    order = 0
    def before_fit(self, learn): pass
    def after_fit(self, learn): pass
    def before_epoch(self, learn): pass
    def after_epoch(self, learn): pass
    def before_batch(self, learn): pass
    def after_batch(self, learn): pass
    def after_predict(self, learn): pass
    def after_loss(self, learn): pass
    def after_backward(self, learn): pass
    def after_step(self, learn): pass
    def cleanup_fit(self, learn): pass


def run_cbs(cbs, method_nm, learn=None):
    for cb in sorted(cbs, key=attrgetter('order')):
        method = getattr(cb, method_nm, None)
        if method is not None: method(learn)


class CompletionCB(Callback):
    def before_fit(self, learn): self.count = 0
    def after_batch(self, learn): self.count += 1
    def after_fit(self, learn): print(f'Completed {self.count} batches')


class Metric:
    def __init__(self): self.reset()
    def reset(self): self.vals, self.ns = [], []
    def add(self, inp, targ=None, n=1):
        try:
            self.last = self.calc(inp, targ)
            self.vals.append(self.last)
            self.ns.append(n)
        except Exception as e:
            print(f"Error in metric calculation: {e}")
            self.last = 0
            self.vals.append(0)
            self.ns.append(n)
    
    @property
    def value(self):
        if not self.vals: return 0
        ns = tensor(self.ns)
        try:
            return ((tensor(self.vals) * ns).sum() / ns.sum()).item()
        except Exception as e:
            print(f"Error computing metric value: {e}")
            return 0
    
    def calc(self, inp, targ): 
        try:
            return (inp == targ).float().mean()
        except Exception as e:
            print(f"Error in calc method: {e}")
            return 0

def to_cpu(x):
    try:
        if isinstance(x, Mapping): return {k: to_cpu(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)): return type(x)(to_cpu(o) for o in x)
        return x.detach().cpu() if isinstance(x, torch.Tensor) else x
    except Exception as e:
        print(f"Error in to_cpu conversion: {e}")
        return x

class MetricsCB(Callback):
    def __init__(self, *ms, **metrics):
        for m in ms: metrics[m.__class__.__name__] = m
        self.metrics = metrics
        self.all_metrics = copy(metrics)
        self.all_metrics['loss'] = self.loss = Mean()

    def _log(self, d): print(d)
    
    def before_fit(self, learn): learn.metrics = self
    
    def before_epoch(self, learn): 
        [m.reset() for m in self.all_metrics.values()]
    
    def after_epoch(self, learn):
        try:
            log = {k: f'{v.compute():.3f}' for k,v in self.all_metrics.items()}
            log['epoch'] = learn.epoch
            log['train'] = 'train' if learn.training else 'eval'
            self._log(log)
        except Exception as e:
            print(f"Error in after_epoch logging: {e}")
    
    def after_batch(self, learn):
        try:
            x,y,*_ = to_cpu(learn.batch)
            for m in self.metrics.values():
                m.update(to_cpu(learn.preds), y)
            self.loss.update(to_cpu(learn.loss), weight=len(x))
        except Exception as e:
            print(f"Error in after_batch metrics update: {e}")

default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_device(obj, device):
    try:
        if isinstance(obj, (list,tuple)): return [to_device(o,device) for o in obj]
        if isinstance(obj, dict): return {k: to_device(v,device) for k,v in obj.items()}
        return obj.to(device) if hasattr(obj,'to') else obj
    except Exception as e:
        print(f"Error in device transfer: {e}")
        return obj

class DeviceCB(Callback):
    def __init__(self, device=default_device): self.device = device
    def before_fit(self, learn): 
        try:
            learn.model.to(self.device)
        except Exception as e:
            print(f"Error moving model to device: {e}")
    def before_batch(self, learn): 
        try:
            learn.batch = to_device(learn.batch, device=self.device)
        except Exception as e:
            print(f"Error moving batch to device: {e}")

class with_cbs:
    def __init__(self, nm): self.nm = nm
    def __call__(self, f):
        def _f(o, *args, **kwargs):
            try:
                o.callback(f'before_{self.nm}')
                f(o, *args, **kwargs)
                o.callback(f'after_{self.nm}')
            except globals()[f'Cancel{self.nm.title()}Exception']: pass
            except Exception as e:
                print(f"Error in {self.nm}: {e}")
            finally: 
                o.callback(f'cleanup_{self.nm}')
        return _f

class Learner:
    def __init__(self, model, dls=None, loss_func=F.mse_loss, lr=0.1, cbs=None, opt_func=optim.Adam):
        self.model = model
        
        if dls is None:
            self.dls = {'train': None, 'valid': None}
        elif isinstance(dls, dict):
            self.dls = dls
        elif isinstance(dls, (list, tuple)):
            self.dls = {'train': dls[0], 'valid': dls[1] if len(dls) > 1 else None}
        else:
            self.dls = {'train': dls, 'valid': None}
            
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
        try:
            for self.iter,self.batch in enumerate(self.dl):
                self._one_batch()
        except Exception as e:
            print(f"Error in epoch iteration: {e}")

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
                with torch.no_grad(): 
                    self.one_epoch(False)

    def fit(self, n_epochs=1, train=True, valid=True, cbs=None, lr=None):
        cbs = fc.L([] if cbs is None else cbs)
        for cb in cbs: self.cbs.append(cb)
        try:
            self.n_epochs = n_epochs
            self.epochs = range(n_epochs)
            if lr is None: lr = self.lr
            self.opt = self.opt_func(self.model.parameters(), lr)
            self._fit(train, valid)
        except Exception as e:
            print(f"Error during training: {e}")
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
    
    def predict(self, learn): 
        try:
            learn.preds = learn.model(*learn.batch[:self.n_inp])
        except Exception as e:
            print(f"Error in prediction: {e}")
            learn.preds = None
            
    def get_loss(self, learn): 
        try:
            if learn.preds is not None:
                learn.loss = learn.loss_func(learn.preds, *learn.batch[self.n_inp:])
            else:
                learn.loss = torch.tensor(0.0, requires_grad=True)
        except Exception as e:
            print(f"Error computing loss: {e}")
            learn.loss = torch.tensor(0.0, requires_grad=True)
            
    def backward(self, learn): 
        try:
            learn.loss.backward()
        except Exception as e:
            print(f"Error in backward pass: {e}")
            
    def step(self, learn): 
        try:
            learn.opt.step()
        except Exception as e:
            print(f"Error in optimizer step: {e}")
            
    def zero_grad(self, learn): 
        try:
            learn.opt.zero_grad()
        except Exception as e:
            print(f"Error in zero_grad: {e}")

class ProgressCB(Callback):
    order = MetricsCB.order + 1
    
    def __init__(self, plot=False): 
        self.plot = plot
        
    def before_fit(self, learn):
        try:
            learn.epochs = self.mbar = master_bar(learn.epochs)
            self.first = True
            if hasattr(learn, 'metrics'): learn.metrics._log = self._log
            self.losses, self.val_losses = [], []
        except Exception as e:
            print(f"Error initializing progress bar: {e}")
            
    def _log(self, d):
        try:
            if self.first:
                self.mbar.write(list(d), table=True)
                self.first = False
            self.mbar.write(list(d.values()), table=True)
        except Exception as e:
            print(f"Error logging progress: {e}")
            
    def before_epoch(self, learn): 
        try:
            learn.dl = progress_bar(learn.dl, leave=False, parent=self.mbar)
        except Exception as e:
            print(f"Error setting up epoch progress bar: {e}")
            
    def after_batch(self, learn):
        try:
            if hasattr(learn, 'loss'):
                learn.dl.comment = f'{learn.loss:.3f}'
                if self.plot and learn.training:
                    self.losses.append(learn.loss.item())
        except Exception as e:
            print(f"Error updating batch progress: {e}")
            
    def after_epoch(self, learn):
        try:
            if not learn.training and self.plot:
                self.val_losses.append(learn.metrics.all_metrics['loss'].compute())
                self.mbar.update_graph([self.losses, self.val_losses])
        except Exception as e:
            print(f"Error updating epoch progress: {e}")

class LRFinderCB(Callback):
    def __init__(self, gamma=1.3, max_mult=3): 
        self.gamma = gamma
        self.max_mult = max_mult
        
    def before_fit(self, learn):
        try:
            self.sched = ExponentialLR(learn.opt, self.gamma)
            self.lrs,self.losses = [],[]
            self.min = math.inf
        except Exception as e:
            print(f"Error initializing LR finder: {e}")
            
    def after_batch(self, learn):
        try:
            if not learn.training: raise CancelEpochException()
            self.lrs.append(learn.opt.param_groups[0]['lr'])
            loss = to_cpu(learn.loss).item()
            self.losses.append(loss)
            if loss < self.min: self.min = loss
            if math.isnan(loss) or loss > self.min * self.max_mult:
                raise CancelFitException()
            self.sched.step()
        except CancelEpochException:
            raise
        except CancelFitException:
            raise
        except Exception as e:
            print(f"Error in LR finder batch processing: {e}")
            
    def cleanup_fit(self, learn):
        try:
            plt.plot(self.lrs, self.losses)
            plt.xscale('log')
            plt.xlabel('Learning Rate')
            plt.ylabel('Loss')
            plt.show()
        except Exception as e:
            print(f"Error plotting LR finder results: {e}")
