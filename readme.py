import os
import cv2 as cv
import torch
import numpy as np
from scipy.io import loadmat

def load_metadata(meta_data_path, image_folder):
  
    pairs = []
    labels = []
    
    # Iterate over each subfolder in the image folder
    for relation_folder in os.listdir(image_folder):
        relation_path = os.path.join(image_folder, relation_folder)
        # Corresponding .mat file
        mat_file = os.path.join(meta_data_path, f"{relation_folder}.mat")
        
        data = loadmat(mat_file)['pairs']
        for pair in data:
                # Extract image filenames and construct full paths
                img1 = os.path.join(relation_path, str(pair[2][0]))
                img2 = os.path.join(relation_path, str(pair[3][0]))
                label = pair[1][0]  # 1 for similar, 0 for dissimilar
                
                pairs.append((img1, img2))
                labels.append(label)
    return pairs, labels
   
def preprocess_image(image_path):
    """
    Preprocess an image by resizing and normalizing.
    """
    image = cv.imread(image_path)
 
    image_resized = cv.resize(image, (224, 224))
    image_normalized = image_resized / 255.0  # Normalize to [0, 1]
    image_transposed = np.transpose(image_normalized, (2, 0, 1))  # Change to (C, H, W) format
    return torch.tensor(image_transposed, dtype=torch.float32)

def loadmat_data(meta_data_path, image_folder):
    pairs, labels = load_metadata(meta_data_path, image_folder)
  
    preprocessed_pairs = []
    for img1_path, img2_path in pairs:

            img1 = preprocess_image(img1_path)
            img2 = preprocess_image(img2_path)
            preprocessed_pairs.append((img1, img2))

    # Convert labels to numpy array first, then to a torch tensor
    labels_np = np.array(labels, dtype=np.float32)
    return preprocessed_pairs, torch.tensor(labels_np, dtype=torch.float32)

# Example usage:


