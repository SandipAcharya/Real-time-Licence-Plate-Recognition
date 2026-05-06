import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Paths to your original folders
image_folder = 'images'
label_folder = 'labels'

# Paths to the new folders where the train/test/val splits will be stored
output_base = '../NUM_Plate/'  # Example: 'data_split'
train_output = os.path.join(output_base, 'train')
test_output = os.path.join(output_base, 'test')
val_output = os.path.join(output_base, 'val')

# Create subfolders for images and labels in train, test, val
for split in ['train', 'test', 'val']:
    os.makedirs(os.path.join(output_base, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_base, split, 'labels'), exist_ok=True)

# Get list of all image filenames (assuming .jpg or .png format)
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

# Split the dataset into train, test, validation sets (e.g., 70% train, 20% val, 10% test)
train_files, test_val_files = train_test_split(image_files, test_size=0.3, random_state=42)
val_files, test_files = train_test_split(test_val_files, test_size=0.33, random_state=42)  # 0.33 * 30% = 10%

# Function to move files into their respective folders
def move_files(file_list, split_type):
    for file in file_list:
        img_path = os.path.join(image_folder, file)
        label_file = file.replace('.jpg', '.txt')
        label_path = os.path.join(label_folder, label_file)

        # Destination directories for images and labels
        img_dest = os.path.join(output_base, split_type, 'images', file)
        label_dest = os.path.join(output_base, split_type, 'labels', label_file)

        # Copy the files to their respective directories
        if os.path.exists(img_path) and os.path.exists(label_path):
            shutil.copy2(img_path, img_dest)
            shutil.copy2(label_path, label_dest)
        else:
            print(f"Missing image or label for {file}, skipping.")

# Move the files for each split
move_files(train_files, 'train')
move_files(val_files, 'val')
move_files(test_files, 'test')

print("Dataset has been split and files moved to respective folders.")
