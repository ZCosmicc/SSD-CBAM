import os
import torch
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
from collections import defaultdict
from pprint import PrettyPrinter

from model import SSD300
from datasets import PascalVOCDataset
from utils import calculate_mAP, find_jaccard_overlap, label_map, rev_label_map

# Setup
data_folder = '/kaggle/working/SSD-CBAM/TextileDefectDetectionReorganizedVOC'
n_splits = 5
keep_difficult = True
batch_size = 16
workers = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pp = PrettyPrinter()

use_cbam = True  # Change to False when evaluating the plain SSD model

# Load all data
print("Loading dataset for fold splits...")
full_dataset = PascalVOCDataset(data_folder, split='TRAIN', keep_difficult=keep_difficult)
indices = list(range(len(full_dataset)))
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)


def calculate_precision_recall(det_boxes, det_labels, true_boxes, true_labels, true_difficulties):
    classwise_tp = defaultdict(int)
    classwise_fp = defaultdict(int)
    classwise_fn = defaultdict(int)

    overall_TP = 0
    overall_FP = 0
    overall_FN = 0

    n_classes = len(label_map)

    for class_id in range(1, n_classes):  # skip background
        for det_b, det_l, true_b, true_l, true_d in zip(det_boxes, det_labels, true_boxes, true_labels, true_difficulties):
            det_mask = det_l == class_id
            true_mask = true_l == class_id

            det_b_class = det_b[det_mask]
            true_b_class = true_b[true_mask]
            true_d_class = true_d[true_mask]

            if true_b_class.size(0) == 0:
                classwise_fp[class_id] += det_b_class.size(0)
                overall_FP += det_b_class.size(0)
                continue

            detected = torch.zeros(len(true_b_class), dtype=torch.uint8).to(device)

            for db in det_b_class:
                ious = find_jaccard_overlap(db.unsqueeze(0), true_b_class)

                if ious.size(1) == 0:
                    classwise_fp[class_id] += 1
                    overall_FP += 1
                    continue

                iou_max, iou_max_idx = ious.max(dim=1)

                if iou_max.item() > 0.5:
                    if detected[iou_max_idx] == 0:
                        classwise_tp[class_id] += 1
                        overall_TP += 1
                        detected[iou_max_idx] = 1
                    else:
                        classwise_fp[class_id] += 1
                        overall_FP += 1
                else:
                    classwise_fp[class_id] += 1
                    overall_FP += 1

            fn = (1 - detected).sum().item()
            classwise_fn[class_id] += fn
            overall_FN += fn

    precision_recall_per_class = {}
    for c in range(1, n_classes):
        TP = classwise_tp[c]
        FP = classwise_fp[c]
        FN = classwise_fn[c]

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        precision_recall_per_class[rev_label_map[c]] = {
            'precision': precision,
            'recall': recall,
            'TP': TP,
            'FP': FP,
            'FN': FN
        }

    overall_precision = overall_TP / (overall_TP + overall_FP) if (overall_TP + overall_FP) > 0 else 0.0
    overall_recall = overall_TP / (overall_TP + overall_FN) if (overall_TP + overall_FN) > 0 else 0.0

    return precision_recall_per_class, {
        'precision': overall_precision,
        'recall': overall_recall,
        'TP': overall_TP,
        'FP': overall_FP,
        'FN': overall_FN
    }


def evaluate_fold(fold, val_indices):
    print(f"\n--- Evaluating Fold {fold + 1} ---")

    model_filename = f'ssd_cbam_fold{fold+1}.pth' if use_cbam else f'ssd_fold{fold+1}.pth'
    checkpoint_path = os.path.join('weights', model_filename)

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found for fold {fold}: {checkpoint_path}")
        return

    # Load model
    print("Loading model...")
    model = SSD300(n_classes=len(label_map), use_cbam=use_cbam)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model = model.to(device)
    model.eval()

    # Prepare validation data
    val_dataset = Subset(full_dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=full_dataset.collate_fn, num_workers=workers)

    # Run evaluation
    det_boxes = []
    det_labels = []
    det_scores = []
    true_boxes = []
    true_labels = []
    true_difficulties = []

    with torch.no_grad():
        for images, boxes, labels, difficulties in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)

            predicted_locs, predicted_scores = model(images)

            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(
                predicted_locs, predicted_scores, min_score=0.3, max_overlap=0.35, top_k=200
            )

            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

    # Metrics
    APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
    pr_results, overall_pr = calculate_precision_recall(det_boxes, det_labels, true_boxes, true_labels, true_difficulties)

    print("\nAverage Precisions (AP) per class:")
    pp.pprint(APs)

    print("\nMean Average Precision (mAP): %.3f" % mAP)

    print("\nPrecision and Recall per class:")
    pp.pprint(pr_results)

    print("\nOverall Precision and Recall:")
    pp.pprint(overall_pr)


if __name__ == '__main__':
    for fold, (_, val_idx) in enumerate(kf.split(indices)):
        evaluate_fold(fold, val_idx)
