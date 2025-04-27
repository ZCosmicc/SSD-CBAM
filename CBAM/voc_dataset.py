# File: voc_dataset.py (experimental using voc 2007 dataset to check which one is actually broken the code or dataset)

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from utils import transform, label_map  # Use your existing transform and label_map

class PascalVOC2007Dataset(Dataset):
    def __init__(self, root, split='test', keep_difficult=False):
        self.root = root
        self.split = split
        self.keep_difficult = keep_difficult

        split_path = os.path.join(root, 'ImageSets', 'Layout', split + '.txt')
        with open(split_path) as f:
            self.image_ids = [line.strip() for line in f]

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        image_path = os.path.join(self.root, 'JPEGImages', image_id + '.jpg')
        annotation_path = os.path.join(self.root, 'Annotations', image_id + '.xml')

        image = Image.open(image_path).convert('RGB')
        target = self.parse_annotation(annotation_path)

        boxes = torch.FloatTensor(target['boxes'])
        labels = torch.LongTensor(target['labels'])
        difficulties = torch.ByteTensor(target['difficulties'])

        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.image_ids)

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []
        difficulties = []

        for object in root.iter('object'):
            difficult = int(object.find('difficult').text == '1')
            label = object.find('name').text.lower().strip()
            if label not in label_map:
                continue

            bbox = object.find('bndbox')
            xmin = int(float(bbox.find('xmin').text)) - 1
            ymin = int(float(bbox.find('ymin').text)) - 1
            xmax = int(float(bbox.find('xmax').text)) - 1
            ymax = int(float(bbox.find('ymax').text)) - 1

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_map[label])
            difficulties.append(difficult)

        return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}

    def collate_fn(self, batch):
        images = []
        boxes = []
        labels = []
        difficulties = []

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties
