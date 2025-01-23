import os
import random
import shutil
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from typing import Optional, List, Tuple
import numpy as np

def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    inputs, labels = zip(*batch)
    inputs = torch.stack(inputs)
    labels = torch.tensor(labels, dtype=torch.long)
    return inputs, labels

class CustomDataset:
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
        self.label_to_idx = {label: idx for idx, label in enumerate(set(y))}
        self.y = [self.label_to_idx[label] for label in y]
        self.cache = {}

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        image_path, label = self.x[i], self.y[i]
        try:
            if image_path in self.cache:
                image = self.cache[image_path]
            else:
                image = Image.open(image_path).convert('RGB')
                if len(self.cache) < 1000:
                    self.cache[image_path] = image

            if self.transform:
                image = self.transform(image)
            return image, label
        except (UnidentifiedImageError, OSError, Exception) as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None

class Sampler:
    def __init__(self, ds, shuffle=False, seed=None):
        self.n = len(ds)
        self.shuffle = shuffle
        self.seed = seed
        self._rng = np.random.RandomState(seed) if seed is not None else np.random

    def __iter__(self):
        indices = np.arange(self.n)
        if self.shuffle:
            self._rng.shuffle(indices)
        return iter(indices)

class BatchSampler:
    def __init__(self, sampler, batch_size, drop_last=False):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

class DataLoader:
    def __init__(self, ds, batch_sampler, collate_fn=None, num_workers=4, prefetch_factor=2):
        self.ds = ds
        self.batch_sampler = batch_sampler
        self.collate_fn = collate_fn or (lambda x: x)
        self.batch_size = batch_sampler.batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.queue_size = max(1, num_workers * prefetch_factor)

        if num_workers > 0:
            self.thread_pool = ThreadPoolExecutor(max_workers=num_workers)
            self.data_queue = queue.Queue(maxsize=self.queue_size)
        else:
            self.thread_pool = None
            self.data_queue = None

    def _load_batch(self, batch_indices):
        batch = []
        for i in batch_indices:
            item = self.ds[i]
            if item is not None:
                batch.append(item)
        if batch:
            return self.collate_fn(batch)
        return None, None

    def _prefetch_worker(self, batch_indices):
        try:
            batch_data = self._load_batch(batch_indices)
            if batch_data[0] is not None:
                self.data_queue.put(batch_data)
        except Exception as e:
            print(f"Error in worker thread: {str(e)}")

    def __iter__(self):
        if self.num_workers == 0:
            for batch_indices in self.batch_sampler:
                batch_data = self._load_batch(batch_indices)
                if batch_data[0] is not None:
                    yield
