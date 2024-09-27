import os
from PIL import image
import numpy as np

# root directory of tumor dataset
root_dir = r'C:\Users\Ara\Desktop\brain_tumor_dataset'

for image_class in os.listdir(data_dr):
    # path to categorized images
    class_dir = os.path.join(root_dir, image_class)

    if os.path.isdir(class_dir):
        if image_class == "yes":
            for images in os.listdir(image_class):
                yes_image = os.path.join(class_dir, images)
                image_processing = Image.open(yes_image)
                yes
                
            
            
            

    
    


