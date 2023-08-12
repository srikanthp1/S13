import torch
from torch.optim.lr_scheduler import OneCycleLR

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

import config
import torch
import torch.optim as optim

from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    print('step started ')
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)


model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
print('model loaded')
optimizer = optim.Adam(
    model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
)
from torch.optim.lr_scheduler import OneCycleLR
train_loader, test_loader, train_eval_loader = get_loaders(
    train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
)

EPOCHS = 15
steps_per_epoch=len(train_loader)
scheduler = OneCycleLR(
        optimizer,
        max_lr=1E-3,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=5/EPOCHS,
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy='linear'
    )
loss_fn = YoloLoss()
scaler = torch.cuda.amp.GradScaler()
print('loss scaler set')



if config.LOAD_MODEL:
    load_checkpoint(
        config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
    )
print('check point set')

scaled_anchors = (
    torch.tensor(config.ANCHORS)
    * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
).to(config.DEVICE)

# Initialize your model, optimizer, and other training setup

# Create a list to store learning rates
learning_rates = []

# Assuming you have a training loop
for epoch in range(20):
    # Train your model
    
    # Update the learning rate using one-cycle policy
    for i in range(steps_per_epoch):
        scheduler.step()
    
    # Get the current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    # Append the current learning rate to the list
    learning_rates.append(current_lr*1000000)
for epoch in range(20):
    learning_rates.append((current_lr-(0.2*current_lr))*1000000)
# Save the list of learning rates to a text file
with open('learning_rates.txt', 'w') as f:
    for lr in learning_rates:
        f.write(f"{lr}\n")

