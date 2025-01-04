import cv2
import numpy as np
import os
from pathlib import Path

def augment_image_with_affine_transformations(image_path, output_dir, num_augments=10):
    """
    Apply affine transformations to an image and save augmented images.
    
    Args:
        image_path (str): Path to the original image.
        output_dir (str): Directory to save the augmented images.
        num_augments (int): Number of augmented images to generate.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    h, w = image.shape[:2]
    print(f"Image dimensions (h, w): ({h}, {w})")

    for i in range(num_augments):
        # Randomize affine transformation parameters
        angle = np.random.uniform(-30, 30)  # Random rotation angle
        scale = np.random.uniform(0.8, 1.2)  # Random scaling
        tx = np.random.uniform(-0.1 * w, 0.1 * w)  # Random translation in x
        ty = np.random.uniform(-0.1 * h, 0.1 * h)  # Random translation in y

        # Compute rotation matrix
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        # Add translation to the transformation matrix
        rotation_matrix[0, 2] += tx
        rotation_matrix[1, 2] += ty

        # Apply affine transformation
        augmented_image = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Save augmented image
        base_name = Path(image_path).stem
        augmented_image_path = os.path.join(output_dir, f"{base_name}_aug_{i}.jpg")
        cv2.imwrite(augmented_image_path, augmented_image)
        print(f"Saved augmented image: {augmented_image_path}")

def augment_images_in_directory(input_dir, output_dir, num_augments=10):
    """
    Apply affine transformations to all images in a directory and save augmented images.
    
    Args:
        input_dir (str): Directory containing the original images.
        output_dir (str): Directory to save the augmented images.
        num_augments (int): Number of augmented images to generate per image.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_file in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_file)
        if os.path.isfile(image_path):
            print(f"Processing image: {image_path}")
            augment_image_with_affine_transformations(image_path, output_dir, num_augments)

# Paths
input_dir = r"C:\Users\hp\OneDrive\Desktop\Mua_img"  # Replace with your dataset folder path
output_dir = r"C:\Users\hp\OneDrive\Desktop\Mua_imgs"  # Replace with your desired output folder path

# Augment the images in the directory
augment_images_in_directory(input_dir, output_dir, num_augments=10)