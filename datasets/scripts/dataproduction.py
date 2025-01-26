import os
import sys
import cv2

from src.preprocessing.prepare_image import ImagePreparer

# Define paths
file_path = "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/datasets/images/scans/"
save_path = "/Users/rileymcnamara/CODE/2025/Australian-Medical-Form-Scanner/datasets/images/prepped/"

# Ensure the save path exists
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Function to preprocess a single image
def preprocess_image(image_path, output_path):
    
    prepr = ImagePreparer(image_path)
    prepd = prepr.prepare_form()

    # Save the processed image
    file_name = os.path.basename(image_path)
    output_file_path = os.path.join(output_path, file_name)
    cv2.imwrite(output_file_path, prepd)

    print(f"Processed and saved: {output_file_path}")

# Process all images in the folder
for file_name in os.listdir(file_path):
    if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
        full_file_path = os.path.join(file_path, file_name)
        preprocess_image(full_file_path, save_path)
