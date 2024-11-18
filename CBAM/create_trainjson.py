import os
import json

# Set the path to your image folder
image_folder = r"C:\Users\Nameless\Downloads\Compressed\ZJU-Leaper-VOC\JPEGImages"

# Get all image files in the folder
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# Sort the file paths (optional, but makes the output consistent)
image_files.sort()

# Write the list to a JSON file
with open("TRAIN_images.json", "w") as f:
    json.dump(image_files, f, indent=2)

print(f"Created TRAIN_images.json with {len(image_files)} image paths.")