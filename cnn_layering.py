import torch
import torch.nn as nn
import torch.nn.functional as F

class TumorNeuralNetwork(nn.Module):
    def __init__(self):
        super(TumorNeuralNetwork, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # Add batch normalization after conv1
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)  # Add batch normalization after conv2
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)  # Add batch normalization after conv3
        self.pool3 = nn.MaxPool2d(2, 2)

        # Manually calculated flatten size for input size (200x200)
        self.fc1 = nn.Linear(24 * 24 * 256, 1024)
        self.drop1 = nn.Dropout(p=0.4)

        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.drop2 = nn.Dropout(p=0.4)

        self.fc3 = nn.Linear(in_features=1024, out_features=1)  # Binary classification (tumor/no tumor)

    def forward(self, x):
        # Pass through convolutional layers with ReLU and pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.drop1(x)

        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        # Binary classification with sigmoid activation
        x = torch.sigmoid(self.fc3(x))

        return x
