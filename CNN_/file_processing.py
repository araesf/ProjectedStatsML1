import os
import cv2

# root directory of tumor dataset
root_dir = r'C:\Users\Ara\Desktop\brain_tumor_dataset'

# initialize lists to hold labels and training data
labels = ["withTumor", "noTumor"]
training = []
img_size = 200

# use OpenCV library for image processing
def createTrainingData():
    for label in labels:
        class_dir = os.path.join(root_dir, label)
        for brain_scan in label:
            cv_image = cv2.imread(os.path.join(class_dir, brain_scan))
            modified_cv_image = cv2.resize(cv.image, (img_size, img_size))
            training.append([modified_cv_image, label])
            
createTrainingData()

        
                
                
    
        
    


