from utils import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter
from model import SSD300
from collections import defaultdict

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Number of classes
n_classes = len(label_map)

rev_label_map = {v: k for k, v in label_map.items()}


# Parameters
#data_folder = '/kaggle/working/SSD-CBAM/TextileDefectDetectionReorganizedVOC'
data_folder = '/kaggle/working/SSD-CBAM/SSD-ReorganizedVOC8010'
keep_difficult = True
batch_size = 16
workers = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cbam = True  # Match this to training config
checkpoint = './results/model_weights_epoch_99.pth'

# Load model
print(f"Initializing SSD300 with CBAM {'enabled' if use_cbam else 'disabled'}")
model = SSD300(n_classes=n_classes, use_cbam=use_cbam)
model.load_state_dict(torch.load(checkpoint), strict=False)
model = model.to(device)
model.eval()

# Load test data
test_dataset = PascalVOCDataset(data_folder, split='test', keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)


def calculate_precision_recall(det_boxes, det_labels, true_boxes, true_labels, true_difficulties):
    from collections import defaultdict

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
                # No ground truth, all detections are false positives
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

    # Per-class metrics
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

    # Overall metrics
    overall_precision = overall_TP / (overall_TP + overall_FP) if (overall_TP + overall_FP) > 0 else 0.0
    overall_recall = overall_TP / (overall_TP + overall_FN) if (overall_TP + overall_FN) > 0 else 0.0

    return precision_recall_per_class, {'precision': overall_precision, 'recall': overall_recall, 'TP': overall_TP, 'FP': overall_FP, 'FN': overall_FN}


def evaluate(test_loader, model):
    """
    Evaluate SSD model on test dataset.
    """
    model.eval()

    det_boxes = []
    det_labels = []
    det_scores = []
    true_boxes = []
    true_labels = []
    true_difficulties = []

    with torch.no_grad():
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)

            predicted_locs, predicted_scores = model(images)

            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(
                predicted_locs, predicted_scores,
                min_score=0.3, max_overlap=0.35, top_k=100
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

        # Compute mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

        # Compute Precision/Recall
        pr_results, overall_pr = calculate_precision_recall(
            det_boxes, det_labels, true_boxes, true_labels, true_difficulties
        )

    # Print results
    print("\nAverage Precisions (AP) per class:")
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)

    print("\nPrecision and Recall per class:")
    pp.pprint(pr_results)

    print("\nOverall Precision and Recall:")
    pp.pprint(overall_pr)

    return mAP, APs, pr_results, overall_pr

if __name__ == '__main__':
    evaluate(test_loader, model)
