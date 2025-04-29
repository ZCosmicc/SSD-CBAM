# File: eval_voc.py

import torch
from tqdm import tqdm
from pprint import PrettyPrinter

from voc_dataset import PascalVOC2007Dataset
from model import SSD300
from utils import label_map, rev_label_map, calculate_mAP, transform_train, transform_test  # import your existing utils

# Pretty print setup
pp = PrettyPrinter()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
voc_root = '/kaggle/input/voc2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007'  # <- adjust this based on your Kaggle dataset
split = 'test'  # 'train', 'trainval', or 'val' depending on what you want

# Load dataset
test_dataset = PascalVOC2007Dataset(root=voc_root, split=split, keep_difficult=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=16, shuffle=False, collate_fn=test_dataset.collate_fn
)

# Quick check dataset
from collections import Counter

print(f"Number of images: {len(test_dataset)}")

all_labels = []

for i in range(min(100, len(test_dataset))):
    _, _, labels, _ = test_dataset[i]
    all_labels.extend(labels.tolist())

label_counter = Counter(all_labels)

print("\nLabel distribution (label IDs):")
for label_id, count in sorted(label_counter.items()):
    print(f"Label {label_id}: {count} instances")

try:
    print("\nLabel distribution (class names):")
    for label_id, count in sorted(label_counter.items()):
        label_name = rev_label_map[label_id]
        print(f"{label_name}: {count} instances")
except NameError:
    print("(rev_label_map not available)")

# Load model
print(f"\nInitializing SSD300 Model")
n_classes = len(label_map)
model = SSD300(n_classes=n_classes, use_cbam=True)  # Set use_cbam=True if you want
checkpoint_path = './voc_results/model_weights_epoch_149.pth'  # <- Adjust if needed (based on how many epoch it is trained)
model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
model = model.to(device)
model.eval()

# Evaluate
def evaluate(test_loader, model):
    model.eval()

    det_boxes = []
    det_labels = []
    det_scores = []
    true_boxes = []
    true_labels = []
    true_difficulties = []

    with torch.no_grad():
        for images, boxes, labels, difficulties in tqdm(test_loader, desc='Evaluating'):
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

    # Results
    print("\nAverage Precisions (AP) per class:")
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)

if __name__ == '__main__':
    evaluate(test_loader, model)
