import torch.nn as nn
import torch.nn.functional as F

class TumorNeuralNetwork(nn.Module):
    def __init__(self):
        super(TumorNeuralNetwork, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2) 

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Flatten Layer
        self.flatten = nn.Flatten()

        # Fully Connected Layers (with 20x20 approximation)
        self.fc1 = nn.Linear(in_features=20 * 20 * 256, out_features=1024)
        self.drop1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.drop2 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(in_features=1024, out_features=10)  # Adjust based on number of classes
        self.drop3 = nn.Dropout(p=0.3)

    def forward(self, x):
        # Pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Flatten the tensor for the fully connected layers
        x = self.flatten(x)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.drop1(x)

        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.drop3(x)

        return x
