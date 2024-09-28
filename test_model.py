import torch
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO
from cnn_layers import TumorNeuralNetwork  # Your model architecture

# Load the trained model
def load_model(model_path, device):
    model = TumorNeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Preprocess the image: resizing, converting to tensor, and normalizing
def preprocess_image(image, img_size=200):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image = transform(image).unsqueeze(0)
    return image

# Download the image from a URL
def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the image: {e}")
        return None
    except Image.UnidentifiedImageError:
        print("The URL does not point to a valid image file.")
        return None

# Classify if the MRI has a tumor or not
def classify_mri_from_url(url, model, device):
    # Download and preprocess the image
    image = load_image_from_url(url)
    if image is None:
        print("Could not load the image. Please provide a valid image URL.")
        return

    # Preprocess the image
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        print(f"Raw model output: {output.item()}")  # Print raw model output
        
        prediction = torch.round(output.squeeze())  # Binary classification: round to 0 or 1
    return "Tumor Detected" if prediction.item() == 1 else "No Tumor"


if __name__ == "__main__":
    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model_path = 'best_model.pth'  # Path to the saved model
    model = load_model(model_path, device)

    # Get the MRI image URL from the user
    image_url = input("Enter the URL of the MRI image: ")
    
    # Classify the MRI
    result = classify_mri_from_url(image_url, model, device)
    
    if result:
        print(result)
