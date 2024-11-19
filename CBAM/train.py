import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *
import sys
torch.autograd.set_detect_anomaly(False)  # Changed to False for speed

# Data parameters
data_folder = '/kaggle/working/SSD-CBAM/TextileDefectDetectionReorganizedVOC'
keep_difficult = True

# Model parameters
n_classes = len(label_map)

# Force CUDA and print GPU info
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
else:
    device = torch.device("cpu")
    print("\nWARNING: CUDA not available, using CPU!")

# Modified learning parameters for testing
checkpoint = None
batch_size = 16
test_epochs = 50
workers = 2  # Reduced workers for better stability
print_freq = 5  # Print more frequently
lr = 1e-3
momentum = 0.9
weight_decay = 5e-4
grad_clip = None

# Modified learning rate decay
decay_lr_at = [test_epochs // 2, int(test_epochs * 0.75)]
decay_lr_to = 0.1

# Enable cudnn benchmarking for speed
cudnn.benchmark = True
cudnn.fastest = True

def main():
    """
    Training with timing measurements and progress tracking.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
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

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Load dataset with memory pinning
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                             batch_size=batch_size, 
                                             shuffle=True,
                                             collate_fn=train_dataset.collate_fn, 
                                             num_workers=workers,
                                             pin_memory=True,
                                             persistent_workers=True)

    # Print test configuration
    print('\nTest Configuration:')
    print(f'Device: {device}')
    print(f'Number of epochs: {test_epochs}')
    print(f'Batch size: {batch_size}')
    print(f'Learning rate: {lr}')
    print(f'Dataset size: {len(train_dataset)} images')
    print(f'Iterations per epoch: {len(train_loader)}')
    print(f'Workers: {workers}\n')

    # Training timing
    total_start_time = time.time()
    epoch_times = []

    try:
        # Epochs
        for epoch in range(start_epoch, test_epochs):
            epoch_start_time = time.time()

            if epoch in decay_lr_at:
                adjust_learning_rate(optimizer, decay_lr_to)

            # One epoch's training
            train(train_loader=train_loader,
                  model=model,
                  criterion=criterion,
                  optimizer=optimizer,
                  epoch=epoch)

            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            
            print(f'\nEpoch {epoch + 1} completed in {epoch_time:.2f} seconds')
            print(f'Average batch time: {epoch_time/len(train_loader):.3f} seconds')
            
            # Optional: Save checkpoint only at last epoch to save time
            if epoch == test_epochs - 1:
                save_checkpoint(epoch, model, optimizer)

        # Final statistics
        total_time = time.time() - total_start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        
        print('\nTraining Complete!')
        print(f'Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)')
        print(f'Average epoch time: {avg_epoch_time:.2f} seconds ({avg_epoch_time/60:.2f} minutes)')
        
        # Estimate full training time
        iterations_per_epoch = len(train_loader)
        full_epochs = 120000 // iterations_per_epoch
        estimated_full_time = avg_epoch_time * full_epochs
        
        print(f'\nEstimated time for full training (120k iterations):')
        print(f'- Approximately {full_epochs} epochs')
        print(f'- {estimated_full_time/3600:.2f} hours ({estimated_full_time/86400:.2f} days)')

    except KeyboardInterrupt:
        print('\nTraining interrupted by user!')
        sys.exit(0)

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()
    
    # Progress bar setup
    total_batches = len(train_loader)

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        try:
            data_time.update(time.time() - start)

            # Move to device
            images = images.to(device, non_blocking=True)
            boxes = [b.to(device, non_blocking=True) for b in boxes]
            labels = [l.to(device, non_blocking=True) for l in labels]

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Loss calculation
            if any(len(b) > 0 for b in boxes):
                loss = criterion(predicted_locs, predicted_scores, boxes, labels)
            else:
                loss = torch.tensor(0.0, requires_grad=True).to(device)

            # Backward prop.
            optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
            loss.backward()

            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)

            optimizer.step()

            # Update metrics
            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            # Print status and progress
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

    print()  # New line after epoch
    del predicted_locs, predicted_scores, images, boxes, labels

if __name__ == '__main__':
    main()
