data_dir = ''
train_dir = "{data_dir}/train"
valid_dir = "{data_dir}/valid"
test_dir = "{data_dir}/test"


input_size = (128,128)
num_classes = 15

import os

if not os.path.exists('models'):
    os.makedirs('models')

MODEL_SAVE_PATH = 'models/cnn_model.pth'
batch_size = 64
learning_rate = 0.001
num_epochs = 20
device = "cuda" if __import__('torch').cuda.is_available() else "cpu"

eval_metrics = ["accuracy", "precision", "recall", "f1_score"]

seed = 42
