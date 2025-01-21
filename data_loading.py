
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
                    yield batch_data
        else:
            try:
                futures = []
                for batch_indices in self.batch_sampler:
                    future = self.thread_pool.submit(self._prefetch_worker, batch_indices)
                    futures.append(future)
                    
                    while self.data_queue.qsize() >= self.queue_size:
                        batch_data = self.data_queue.get()
                        yield batch_data

                for _ in futures:
                    if not self.data_queue.empty():
                        batch_data = self.data_queue.get()
                        yield batch_data
                        
            except Exception as e:
                print(f"Error in data loading: {str(e)}")
            finally:
                while not self.data_queue.empty():
                    try:
                        self.data_queue.get_nowait()
                    except queue.Empty:
                        break

    def __len__(self):
        n_samples = len(self.ds)
        if self.batch_sampler.drop_last:
            return n_samples // self.batch_size
        return (n_samples + self.batch_size - 1) // self.batch_size

    @property
    def dataset(self):
        return self.ds

def validate_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except:
        return False

def split_dataset(data_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = []
        for f in os.listdir(class_dir):
            file_path = os.path.join(class_dir, f)
            if os.path.isfile(file_path) and validate_image(file_path):
                images.append(f)
            else:
                print(f"Skipping invalid image: {file_path}")

        if not images:
            print(f"Warning: No valid images found in {class_dir}")
            continue

        train_imgs, temp_imgs = train_test_split(images, test_size=(val_ratio + test_ratio), random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

        for split, img_list in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            for img in img_list:
                src_path = os.path.join(class_dir, img)
                dst_path = os.path.join(split_class_dir, img)
                shutil.copy(src_path, dst_path)

    print(f"Dataset split into train, val, and test sets at: {output_dir}")

def create_dataloaders(output_dir, batch_size=32, num_workers=4):
    train_transform = transforms.Compose([
        transforms.Resize((128,128)),  
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128,128)),  
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    def load_split(split):
        split_dir = os.path.join(output_dir, split)
        image_paths = []
        labels = []
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                if validate_image(img_path):
                    image_paths.append(img_path)
                    labels.append(class_name)
                else:
                    print(f"Skipping invalid image during dataloader creation: {img_path}")
        return image_paths, labels

    train_paths, train_labels = load_split('train')
    val_paths, val_labels = load_split('val')
    test_paths, test_labels = load_split('test')

    train_dataset = CustomDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = CustomDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = CustomDataset(test_paths, test_labels, transform=val_transform)

    train_sampler = BatchSampler(Sampler(train_dataset, shuffle=True, seed=42), batch_size)
    val_sampler = BatchSampler(Sampler(val_dataset, shuffle=False), batch_size)
    test_sampler = BatchSampler(Sampler(test_dataset, shuffle=False), batch_size)

    train_loader = DataLoader(
        train_dataset, 
        train_sampler, 
        collate_fn=collate,
        num_workers=num_workers,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, 
        val_sampler, 
        collate_fn=collate,
        num_workers=num_workers,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        test_dataset, 
        test_sampler, 
        collate_fn=collate,
        num_workers=num_workers,
        prefetch_factor=2
    )

    return train_loader, val_loader, test_loader
