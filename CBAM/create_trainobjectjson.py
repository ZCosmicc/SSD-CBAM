import os
import json
import xml.etree.ElementTree as ET

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    filename = root.find('filename').text
    boxes = []
    labels = []
    difficulties = []
    
    for obj in root.findall('object'):
        difficult = int(obj.find('difficult').text)
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
        difficulties.append(difficult)
    
    return filename, {"boxes": boxes, "labels": labels, "difficulties": difficulties}

# Set paths
xml_dir = r"C:\Users\Nameless\Downloads\Compressed\ZJU-Leaper-VOC\Annotations"
json_file = "TRAIN_object.json"

# Get all XML files
xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

# Create a dictionary to map class names to integer labels
class_to_label = {}
label_counter = 1

# Parse all XML files
annotations = {}
for xml_file in xml_files:
    filename, data = parse_xml(os.path.join(xml_dir, xml_file))
    
    # Convert class names to integer labels
    for i, label in enumerate(data['labels']):
        if label not in class_to_label:
            class_to_label[label] = label_counter
            label_counter += 1
        data['labels'][i] = class_to_label[label]
    
    annotations[filename] = data

# Convert annotations dictionary to list in the order of TRAIN_images.json
with open("TRAIN_images.json", 'r') as f:
    train_images = json.load(f)

ordered_annotations = [annotations[os.path.basename(img_path)] for img_path in train_images]

# Write to JSON file
with open(json_file, 'w') as f:
    json.dump(ordered_annotations, f)

print(f"Created {json_file} with annotations for {len(ordered_annotations)} images.")
print("Class to label mapping:", class_to_label)