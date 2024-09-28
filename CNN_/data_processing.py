import os
import cv2

# root directory of tumor dataset
root_dir = r'C:\Users\Ara\Desktop\brain_tumor_dataset'

# initialize lists to hold labels and training data
labels = ["withTumor", "withoutTumor"]
training = []
IMG_SIZE  = 200

# use OpenCV library for image processing
def createTrainingData():
    for label in labels:
        class_dir = os.path.join(root_dir, label)
        
        # iterate over every image in the path, compressing the image as well
        for brain_scan in os.listdir(class_dir):
            img_path = os.path.join(class_dir, brain_scan)
            cv_image = cv2.imread(img_path)
            modified_cv_image = cv2.resize(cv_image, (IMG_SIZE, IMG_SIZE))

            # add the modified image to training
            if modified_cv_image is not None:
                training.append([modified_cv_image, label])
                print(f'Added scan to training list: {brain_scan}')
    return training
                
    
        
        
    


