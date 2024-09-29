import torch
import torch.nn as nn

class TumorNeuralNetwork(nn.Module):
    def __init__(self):
        super(TumorNeuralNetwork, self).__init__()

        # Convolutional layers with increased capacity, batch normalization, and ReLU
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((2, 2))  # Try larger output size for better feature retention
        )

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),  # Adjusted input size after pooling
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),  # Increased dropout to prevent overfitting
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()  # Binary classification
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
