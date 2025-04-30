import os
import torch
from model import SSD300
from datasets import PascalVOCDataset
from utils import *  
from eval import evaluate  # reuse your existing evaluate()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Settings
data_root = '/kaggle/working/SSD-CBAM/TextileDefectDetectionReorganizedVOC'
n_folds = 5
use_cbam = True
batch_size = 16
num_workers = 8

fold_mAPs = []

for fold in range(n_folds):
    print(f"\n=== Evaluating Fold {fold} ===")

    # Dataset
    test_dataset = PascalVOCDataset(
        root=os.path.join(data_root, f'fold_{fold}'),
        split='test',
        keep_difficult=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=test_dataset.collate_fn, num_workers=num_workers, pin_memory=True
    )

    # Load model
    checkpoint_path = f'fold_{fold}_best.pth'
    model = SSD300(n_classes=len(label_map), use_cbam=use_cbam)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model = model.to(device)

    # Evaluate
    mAP, APs, pr_results, overall_pr = evaluate(test_loader, model)
    fold_mAPs.append(mAP)
    print(f"Fold {fold} mAP: {mAP:.3f}")

# Summary
avg_mAP = sum(fold_mAPs) / len(fold_mAPs)
print("\n===== Cross-Validation Evaluation Complete =====")
print(f"mAPs per fold: {[round(m, 3) for m in fold_mAPs]}")
print(f"Average mAP: {avg_mAP:.3f}")
