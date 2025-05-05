import os
import time
import json
import torch
import random
import shutil
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
from sklearn.model_selection import KFold
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *
from train import train

torch.autograd.set_detect_anomaly(False)

# Parameters
data_folder = '/kaggle/working/SSD-CBAM/TextileDefectDetectionReorganizedVOC'
keep_difficult = True
use_cbam = False

n_classes = len(label_map)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16
test_epochs = 100
workers = 8
print_freq = 5
lr = 1e-3
momentum = 0.9
weight_decay = 5e-4
decay_lr_at = [test_epochs // 2, int(test_epochs * 0.75)]
decay_lr_to = 0.1

def load_full_dataset(split):
    with open(os.path.join(data_folder, f'{split}_images.json')) as f:
        images = json.load(f)
    with open(os.path.join(data_folder, f'{split}_objects.json')) as f:
        objects = json.load(f)
    return images, objects

def save_fold_data(images, objects, split_name, fold):
    os.makedirs(f'./fold_data/fold{fold}', exist_ok=True)
    with open(f'./fold_data/fold{fold}/{split_name}_images.json', 'w') as f:
        json.dump(images, f)
    with open(f'./fold_data/fold{fold}/{split_name}_objects.json', 'w') as f:
        json.dump(objects, f)

def train_one_fold(fold, train_idx, val_idx, all_images, all_objects):
    print(f"\n--- Fold {fold+1} ---")

    train_images = [all_images[i] for i in train_idx]
    train_objects = [all_objects[i] for i in train_idx]
    val_images = [all_images[i] for i in val_idx]
    val_objects = [all_objects[i] for i in val_idx]

    save_fold_data(train_images, train_objects, 'TRAIN', fold)
    save_fold_data(val_images, val_objects, 'TEST', fold)

    model = SSD300(n_classes=n_classes, use_cbam=use_cbam).to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    biases = [p for n, p in model.named_parameters() if p.requires_grad and n.endswith('.bias')]
    not_biases = [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith('.bias')]

    optimizer = torch.optim.SGD(
        [{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
        lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    cudnn.benchmark = True

    train_dataset = PascalVOCDataset(f'./fold_data/fold{fold}', 'TRAIN', keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True, persistent_workers=True)

    for epoch in range(test_epochs):
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)
        train(train_loader, model, criterion, optimizer, epoch)

    # Ensure weights folder exists
    os.makedirs('weights', exist_ok=True)

    # Save model
    torch.save(model.state_dict(), f'weights/ssd_cbam_fold{fold+1}.pth' if use_cbam else f'weights/ssd_fold{fold+1}.pth')
    print(f"Saved model for fold {fold+1}\n")


def main():
    all_images, all_objects = load_full_dataset('TRAIN')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
        train_one_fold(fold, train_idx, val_idx, all_images, all_objects)

if __name__ == '__main__':
    main()
