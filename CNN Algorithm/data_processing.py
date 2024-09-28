import os
import cv2

class TumorDataProcessor:
    def __init__(self, root_dir, labels, img_size=200):
        """
        Initialize the TumorDataProcessor with dataset directory, labels, and image size.
        """
        self.root_dir = root_dir
        self.labels = labels
        self.img_size = img_size
        self.training_data = []

    def create_training_data(self):
        """
        Loads the images, processes them, and appends them with their respective labels to training_data.
        """
        for label in self.labels:
            class_dir = os.path.join(self.root_dir, label)
            
            # Iterate over every image in the label directory
            for brain_scan in os.listdir(class_dir):
                img_path = os.path.join(class_dir, brain_scan)
                
                # Read and resize the image using OpenCV
                cv_image = cv2.imread(img_path)
                modified_cv_image = cv2.resize(cv_image, (self.img_size, self.img_size))

                # If the image was read and processed correctly, add it to training_data
                if modified_cv_image is not None:
                    self.training_data.append([modified_cv_image, label])
                    print(f'Added scan to training list: {brain_scan}')

    def get_training_data(self):
        """
        Returns the processed training data.
        """
        return self.training_data

if __name__ == "__main__":
    # Define the root directory and labels
    root_dir = r'C:\Users\Ara\Desktop\brain_tumor_dataset'
    labels = ["withTumor", "withoutTumor"]

    # Initialize the processor
    processor = TumorDataProcessor(root_dir=root_dir, labels=labels, img_size=200)
    
    # Create the training data
    processor.create_training_data()
    
    # Retrieve the training data
    training_data = processor.get_training_data()
    print(f'Total training samples: {len(training_data)}')
