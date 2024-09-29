import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

class TumorDataProcessor:
    def __init__(self, root_dir, labels, img_size=200):
        self.root_dir = root_dir
        self.labels = labels
        self.img_size = img_size
        self.training_data = []

    def create_training_data(self):
        for label in self.labels:
            class_dir = os.path.join(self.root_dir, label)
            for brain_scan in os.listdir(class_dir):
                img_path = os.path.join(class_dir, brain_scan)
                cv_image = cv2.imread(img_path)
                modified_cv_image = cv2.resize(cv_image, (self.img_size, self.img_size))
                if modified_cv_image is not None:
                    self.training_data.append([modified_cv_image, label])
                    print(f'Added scan to training list: {brain_scan}')

    def get_training_data(self):
        return self.training_data


def convert_to_tensor(data):
    X = []
    y = []
    for img, label in data:
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = img / 255.0  # Normalize pixel values
        X.append(img)
        y.append(1 if label == "withTumor" else 0)
    
    X_array = np.array(X, dtype=np.float32)
    y_array = np.array(y, dtype=np.float32)

    X_tensor = torch.tensor(X_array)
    y_tensor = torch.tensor(y_array)

    return TensorDataset(X_tensor, y_tensor)


def load_data(root_dir, labels):
    processor = TumorDataProcessor(root_dir=root_dir, labels=labels, img_size=200)
    processor.create_training_data()
    training_data = processor.get_training_data()
    
    train_size = int(0.8 * len(training_data))
    val_size = len(training_data) - train_size
    train_data, val_data = torch.utils.data.random_split(training_data, [train_size, val_size])

    train_dataset = convert_to_tensor(train_data)
    val_dataset = convert_to_tensor(val_data)

    return train_dataset, val_dataset
