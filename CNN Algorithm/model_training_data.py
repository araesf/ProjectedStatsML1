import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from cnn_layers import TumorNeuralNetwork  # Import the model
from data_preparation import load_data  # Import the data loader
import torch.nn as nn

def train_model(train_loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs.squeeze(), labels)
        
        # Backward pass and optimization
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss / len(train_loader)

def validate_model(val_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs.squeeze(), labels)
            running_loss += loss.item()
    
    return running_loss / len(val_loader)

# Ensure 'model' and data loaders are defined before the training loop
if __name__ == "__main__":
    # Load data
    root_dir = r'C:\Users\Ara\Desktop\brain_tumor_dataset'  # Adjust the path
    labels = ["withTumor", "withoutTumor"]

    # Load the train and validation datasets
    train_dataset, val_dataset = load_data(root_dir, labels)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, optimizer, and loss function
    model = TumorNeuralNetwork().to(device)
    
    # Use a lower learning rate and add weight decay (L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)
    
    # Binary Cross-Entropy Loss for binary classification
    criterion = nn.BCELoss()

    # Learning rate scheduler without verbose=True
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # Initialize early stopping parameters
    best_val_loss = float('inf')
    patience = 3
    early_stopping_counter = 0

    # Inside the training loop, manually print the learning rate
    n_epochs = 20
    for epoch in range(n_epochs):
        train_loss = train_model(train_loader, model, criterion, optimizer, device)
        val_loss = validate_model(val_loader, model, criterion, device)

        # Learning rate scheduling based on validation loss
        scheduler.step(val_loss)

        # Print the current learning rate using get_last_lr()
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Learning Rate: {current_lr:.6f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print("Early stopping triggered")
            break
