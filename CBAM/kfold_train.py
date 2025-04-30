import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from datasets import PascalVOCDataset
from model import SSD300, SSD_CBAM  # Adjust if class names are different
from train import train_one_epoch
from eval import evaluate
from utils import transform, MultiBoxLoss

# === Config ===
data_folder = 'data'  # Adjust to your dataset folder
batch_size = 16
num_epochs = 100
num_folds = 5
use_cbam = True  # Set to False to train baseline SSD
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Load Full Dataset (we use 'TRAIN' for both train/val in k-fold) ===
full_dataset = PascalVOCDataset(data_folder=data_folder, split='TRAIN', keep_difficult=False)
indices = list(range(len(full_dataset)))

# === Set up Cross Validation ===
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
    print(f"\n==== Fold {fold + 1} ====\n")

    # Datasets for current fold
    train_dataset = PascalVOCDataset(data_folder, split='TRAIN', keep_difficult=False, indices=train_idx)
    val_dataset = PascalVOCDataset(data_folder, split='TRAIN', keep_difficult=False, indices=val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=train_dataset.collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=val_dataset.collate_fn, num_workers=4)

    # === Initialize Model ===
    model = SSD_CBAM(num_classes=21) if use_cbam else SSD300(num_classes=21)  # Change num_classes if needed
    model = model.to(device)

    # === Loss & Optimizer ===
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    # === Training Loop ===
    for epoch in range(num_epochs):
        print(f'[Fold {fold + 1}][Epoch {epoch + 1}/{num_epochs}]')
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        evaluate(model, val_loader, device)

    # === Save Model ===
    os.makedirs('weights', exist_ok=True)
    model_type = 'ssd_cbam' if use_cbam else 'ssd'
    torch.save(model.state_dict(), f'weights/{model_type}_fold{fold + 1}.pth')
