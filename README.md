# Model Training and Data Loading System for plant-disease-detection

This repository provides an end-to-end solution for training, validating, and testing deep learning models using PyTorch. It includes modules for model training, architecture definition, and dataset loading with data augmentation.The dataloader and learner is created from scratch for customisation. The system achieved **95% accuracy on the test set** when configured as described.

---

## Overview

The project is designed specifically for the **PlantVillage Dataset** and consists of three main files:
1. **`model_training.py`**: Handles the training and validation processes, with support for callbacks and gradient scaling for mixed precision training.
2. **`model_architecture.py`**: Defines the Convolutional Neural Network (CNN) architecture.
3. **`data_loading.py`**: Manages dataset splitting, preprocessing, and loading into PyTorch-compatible DataLoader objects.

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- Pillow

To install the required packages:
```bash
pip install torch torchvision scikit-learn matplotlib Pillow
```

---

## File Descriptions

### `model_training.py`
This script includes:
- **Callback System**: Flexible callbacks like `PrintCallback`, `PlotCallback`, and `TestCallback` for monitoring training and evaluation metrics.
- **Learner Class**: Encapsulates the training, validation, and testing processes.
- **Mixed Precision Support**: Utilizes PyTorch AMP for faster training on compatible GPUs.
- **Metrics Tracking**: Logs loss and accuracy for both training and validation phases.

### `model_architecture.py`
Defines a customizable CNN architecture:
- Convolutional layers with batch normalization and ReLU activation.
- Adaptive average pooling for dimensionality reduction.
- Fully connected layers for classification.

### `data_loading.py`
Manages dataset operations:
- **Dataset Splitting**: Divides data into train, validation, and test sets.
- **Custom Dataset Class**: Loads images and labels with optional data augmentation.
- **Data Augmentation**: Applies transformations like resizing, horizontal flipping, rotation, and normalization.
- **DataLoader**: Efficiently batches data with support for custom collation and shuffling.

---

## How to Use

### Step 1: Load the Dataset
The **PlantVillage Dataset** should already be organized into subdirectories where each subdirectory corresponds to a class label. For example:
```
/plantvillage
    /healthy
        image1.jpg
        image2.jpg
    /diseased
        image3.jpg
        image4.jpg
```

### Step 2: Split the Dataset
Use the `split_dataset` function to create train, validation, and test sets:
```python
from data_loading import split_dataset

split_dataset("/path/to/plantvillage", "/path/to/output", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
```

### Step 3: Create DataLoaders
Generate DataLoaders for the train, validation, and test sets:
```python
from data_loading import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders("/path/to/output", batch_size=32)
```

### Step 4: Define the Model
Initialize the CNN architecture:
```python
from model_architecture import cnn_arch

model = cnn_arch
```

### Step 5: Train the Model
Train the model using the `Learner` class:
```python
from model_training import Learner, PrintCallback, PlotCallback, TestCallback

callbacks = [PrintCallback(), PlotCallback(), TestCallback()]

learner = Learner(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    loss_fn=torch.nn.CrossEntropyLoss(),
    callbacks=callbacks,
)

history = learner.fit(epochs=10)
```

For replicating the **95% test accuracy**, ensure the following:
- Use a batch size of **32**.
- Learning rate set to **0.001**.
- Train the model for **10 epochs** using the default configurations.

---

## Visualizing Metrics
The `PlotCallback` generates plots for loss and accuracy:
- Training vs Validation Loss
- Training vs Validation Accuracy

These plots are displayed after every epoch.

---

## Customization

### Modify the Model
Update the `CNNArchitecture` class in `model_architecture.py` to experiment with different architectures.

### Add New Callbacks
Extend the `Callback` class in `model_training.py` to implement custom behaviors during training.

---

## Error Handling
- Invalid images are skipped during dataset loading and logged to the console.
- Fallbacks are in place for handling empty batches.

---

## License
This project is released under the Apache 2.0 license.

---


