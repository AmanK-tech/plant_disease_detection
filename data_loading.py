
import os
import random
import shutil
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

def collate(batch):
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

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        image_path, label = self.x[i], self.y[i]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label



class Sampler:
    def __init__(self, ds, shuffle=False):
        self.n = len(ds)
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(self.n))
        if self.shuffle:
            random.shuffle(indices)
        return iter(indices)


class BatchSampler:
    def __init__(self, sampler, batch_size, drop_last=False):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in iter(self.sampler):
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch


class DataLoader:
    def __init__(self, ds, batch_sampler, collate_fn=None):
        self.ds = ds
        self.batch_sampler = batch_sampler
        self.collate_fn = collate_fn or (lambda x: x)

    def __iter__(self):
        for batch_indices in self.batch_sampler:
            batch = [self.ds[i] for i in batch_indices]
            yield self.collate_fn(batch)



def split_dataset(data_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    assert abs(train_ratio + val_ratio + test_ratio - 1) < 1e-6

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

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


def create_dataloaders(output_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  
        transforms.ToTensor()
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
                image_paths.append(os.path.join(class_dir, img_file))
                labels.append(class_name)
        return image_paths, labels

    train_paths, train_labels = load_split('train')
    val_paths, val_labels = load_split('val')
    test_paths, test_labels = load_split('test')

    train_dataset = CustomDataset(train_paths, train_labels, transform=transform)
    val_dataset = CustomDataset(val_paths, val_labels, transform=transform)
    test_dataset = CustomDataset(test_paths, test_labels, transform=transform)

    train_sampler = BatchSampler(Sampler(train_dataset, shuffle=True), batch_size)
    val_sampler = BatchSampler(Sampler(val_dataset, shuffle=False), batch_size)
    test_sampler = BatchSampler(Sampler(test_dataset, shuffle=False), batch_size)

    train_loader = DataLoader(train_dataset, train_sampler,collate_fn=collate)
    val_loader = DataLoader(val_dataset, val_sampler,collate_fn=collate)
    test_loader = DataLoader(test_dataset, test_sampler, collate_fn=collate)

    return train_loader, val_loader, test_loader







