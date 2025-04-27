# File: train_voc.py

import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import sys
import os

from model import SSD300, MultiBoxLoss
from voc_dataset import PascalVOC2007Dataset
from utils import *
torch.autograd.set_detect_anomaly(False)

# Data parameters
data_folder = '/kaggle/input/pascal-voc-2007-data-view/VOCdevkit2007/VOC2007/'  # Adjust here
keep_difficult = True

# Model parameters
n_classes = len(label_map)  # Should be 21 for VOC (background + 20 classes)

# CUDA setup
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("\nWARNING: CUDA not available, using CPU!")

# Learning parameters
checkpoint = None
batch_size = 16
epochs = 100
workers = 8
print_freq = 5
lr = 1e-3
momentum = 0.9
weight_decay = 5e-4
grad_clip = None

decay_lr_at = [epochs // 2, int(epochs * 0.75)]
decay_lr_to = 0.1

cudnn.benchmark = True
cudnn.fastest = True

def save_checkpoint(epoch, model, optimizer, save_path='./voc_results'):
    os.makedirs(save_path, exist_ok=True)

    should_save = (epoch == epochs // 2) or (epoch == epochs - 1)

    if should_save:
        checkpoint_path = os.path.join(save_path, f'checkpoint_epoch_{epoch}.pth.tar')
        model_path = os.path.join(save_path, f'model_weights_epoch_{epoch}.pth')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)

        torch.save(model.state_dict(), model_path)

        print(f"\nCheckpoint saved to {checkpoint_path}")
        print(f"Model weights saved to {model_path}")

def main():
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes, use_cbam=False)  # ← Change to True if you want CBAM
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Load dataset
    train_dataset = PascalVOC2007Dataset(root=data_folder, split='trainval', keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               collate_fn=train_dataset.collate_fn,
                                               num_workers=workers,
                                               pin_memory=True,
                                               persistent_workers=True)

    print('\nTraining Configuration:')
    print(f'Device: {device}')
    print(f'Number of epochs: {epochs}')
    print(f'Batch size: {batch_size}')
    print(f'Learning rate: {lr}')
    print(f'Dataset size: {len(train_dataset)} images')
    print(f'Workers: {workers}\n')

    total_start_time = time.time()
    epoch_times = []

    try:
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()

            if epoch in decay_lr_at:
                adjust_learning_rate(optimizer, decay_lr_to)

            train(train_loader, model, criterion, optimizer, epoch)

            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)

            print(f'\nEpoch {epoch + 1} completed in {epoch_time:.2f} seconds')
            print(f'Average batch time: {epoch_time/len(train_loader):.3f} seconds')

            save_checkpoint(epoch, model, optimizer, save_path='./voc_results')

        total_time = time.time() - total_start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        
        print('\nTraining Complete!')
        print(f'Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)')
        print(f'Average epoch time: {avg_epoch_time:.2f} seconds ({avg_epoch_time/60:.2f} minutes)')
        print("\nResults saved in './voc_results' directory.")

    except KeyboardInterrupt:
        print('\nTraining interrupted by user!')
        sys.exit(0)

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()

    total_batches = len(train_loader)

    for i, (images, boxes, labels, _) in enumerate(train_loader):
        try:
            data_time.update(time.time() - start)

            images = images.to(device, non_blocking=True)
            boxes = [b.to(device, non_blocking=True) for b in boxes]
            labels = [l.to(device, non_blocking=True) for l in labels]

            predicted_locs, predicted_scores = model(images)

            if any(len(b) > 0 for b in boxes):
                loss = criterion(predicted_locs, predicted_scores, boxes, labels)
            else:
                loss = torch.tensor(0.0, requires_grad=True).to(device)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)

            optimizer.step()

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            if i % print_freq == 0:
                print(f'\rEpoch: [{epoch}][{i}/{total_batches}] '
                      f'({100. * i / total_batches:.0f}%) '
                      f'Batch Time: {batch_time.val:.3f}s '
                      f'({batch_time.avg:.3f}s) '
                      f'Data Time: {data_time.val:.3f}s '
                      f'Loss: {losses.val:.4f} '
                      f'({losses.avg:.4f})', end='', flush=True)

            start = time.time()

        except Exception as e:
            print(f"\nError in batch {i}: {str(e)}")
            continue

    print()
    del predicted_locs, predicted_scores, images, boxes, labels

if __name__ == '__main__':
    main()
