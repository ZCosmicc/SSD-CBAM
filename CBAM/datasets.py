import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import basic_transform


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    Supports optional indexing for K-Fold Cross-Validation.
    """

    def __init__(self, data_folder, split, keep_difficult=False, indices=None):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        :param indices: optional list of indices to subset the dataset (for k-fold)
        """
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Load full data
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            all_images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            all_objects = json.load(j)

        assert len(all_images) == len(all_objects)

        # Apply index filtering if provided
        if indices is not None:
            self.images = [all_images[i] for i in indices]
            self.objects = [all_objects[i] for i in indices]
        else:
            self.images = all_images
            self.objects = all_objects

    def __getitem__(self, i):
        image = Image.open(self.images[i], mode='r').convert('RGB')

        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each
