import os
import shutil

source_folder = "C:/2data/cholec80/frames"
destination_folder = "C:/1projects/codes/Object_centric/data/Choracic/images/train"

# Walk through the source directory
for root, _, files in os.walk(source_folder):
    # Copy each file to the destination folder
    for file in files:
        source_file_path = os.path.join(root, file)
        # Copy the file to the destination folder
        shutil.copy(source_file_path, destination_folder)

print("All images copied successfully.")