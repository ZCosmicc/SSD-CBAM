# K-Fold SSD Training Script
# Updated train.py with k-fold support for ZJU Leaper dataset

import os
import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader

from datasets import ZJULeaperDataset  # Custom dataset class
from utils import build_ssd, train_one_epoch, evaluate_model  # Assume these are implemented

# Configuration
DATA_ROOT = '/path/to/ZJU-Leaper'  # Update this path
NUM_EPOCHS = 100
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_FOLDS = 5
USE_CBAM = True  # Set to False for baseline SSD
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load full dataset (no transforms yet)
full_dataset = ZJULeaperDataset(root=DATA_ROOT, split='all')
indices = list(range(len(full_dataset)))

# Initialize KFold
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
    print(f"\n=== Fold {fold + 1}/{NUM_FOLDS} ===")

    # Subset datasets
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    # Apply transform (important to do here)
    train_dataset.dataset.set_mode('train')
    val_dataset.dataset.set_mode('test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=full_dataset.collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=full_dataset.collate_fn, num_workers=NUM_WORKERS)

    # Build model
    model = build_ssd(num_classes=full_dataset.num_classes, use_cbam=USE_CBAM)
    model = model.to(DEVICE)

    # Optimizer & loss
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.MultiBoxLoss().to(DEVICE)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

    # Save model after training this fold
    save_path = f'./results/ssd_cbam_{USE_CBAM}_fold{fold + 1}.pth'
    torch.save(model.state_dict(), save_path)

    # Evaluate model
    print("\nEvaluating model on validation set...")
    evaluate_model(model, val_loader, DEVICE, fold=fold + 1)
