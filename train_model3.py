
# -------- INSTALLATION --------

'''
pip install torch torchvision torchaudio
pip install matplotlib
pip install pillow
pip install pycocotools
pip install tqdm
pip install gradio
pip install transformers
pip install huggingface_hubs 
pip install albumentations
'''

# -------- IMPORTS --------
import torch
from torch.utils.data import Dataset, DataLoader
import json

# For visualizing predictions
import torchvision.transforms.functional as F
from matplotlib.patches import Rectangle

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
import torchvision.transforms.functional as F
import random
from pycocotools.coco import COCO
import albumentations as A

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import visualize_samples
from utils import evaluate_mAP
from utils import save_checkpoint

'''
handle COCO style datasets - used for object detection 
masks tell model the precise outline of each mushroom 
Bounding boxes - rectangle around mushroom 
segmentation - exact pixels of mushroom 
image_id - links annotation to correct image 

Resize images - will make everything smoother 
Most images are JPEG - already RGB
Bottleneck is disk space 
Region Proposal Network - "where the obj is"


'''

# ------------------- DEVICE & MEMORY -------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.empty_cache()  # clear any cached memory
torch.backends.cuda.max_split_size_mb = 64  # reduce frag

# -------- TRANSFORMS DEF --------
augmentations = A.Compose([
    A.Affine(
        scale=(0.8, 1.2),
        rotate=(-45, 45),  # safer than full 360
        translate_percent=(-0.15, 0.15),
        fit_output=True,
        p=0.8
    ),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5)
],
bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# -------- CUSTOM DATASET CLASS --------
class MushroomCOCODataset(Dataset):
    def __init__(self, images_dir, annotations_file, augmentations=None, resize=(384, 384)):
        self.images_dir = images_dir
        self.coco = COCO(annotations_file)
        self.augmentations = augmentations
        self.resize = resize

        # Filter images that have at least one annotation
        self.img_ids = [
            img_id for img_id in self.coco.imgs.keys()
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB") # convert to RBG 
        w_original, h_original = image.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids) # returns list of dictionaries 

        
        boxes = [] # for tensor transformation 
        labels = []

        for ann in anns:
            if ann['category_id'] == 0:
                continue  # skip generic / placeholder

            x, y, width, height = ann['bbox']
            if width <= 1 or height <= 1:
                continue

            boxes.append([x, y, x + width, y + height])
            labels.append(1)  # all mushrooms = class 1
        

        if len(boxes) == 0: # skip images with no valid boxes
            raise ValueError(f"Image {img_id} has no valid mushrooms.")

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Resize
        w_new, h_new = self.resize
        scale_x = w_new / w_original
        scale_y = h_new / h_original
        image = image.resize((w_new, h_new), resample=Image.BILINEAR)
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        # Augmentations
        if self.augmentations:
            transformed = self.augmentations(
                image=np.array(image),
                bboxes=boxes.numpy().tolist(),
                labels=labels.numpy().tolist()
            )
            image = transformed['image']
            boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.as_tensor([int(l) for l in transformed['labels']], dtype=torch.int64)

        # Final validation
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        valid = (widths > 1) & (heights > 1)
        boxes = boxes[valid]
        labels = labels[valid]

        # Convert image to tensor
        image = torch.as_tensor(np.array(image), dtype=torch.float32).permute(2,0,1) / 255.0

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        # Sanity check to confirm lengths match
        # assert len(boxes) == len(labels), f"Boxes ({len(boxes)}) and labels ({len(labels)}) mismatch!"

        return image, target


# -------- DATALOADING --------
best_val_loss = float('inf')
checkpoint_interval = 100
batch_size = 1     # smaller to avoid memory issues on 4GB
accum_steps = 1

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

train_dataset = MushroomCOCODataset(
    images_dir="/home/matson/M18K_dataset/M18KV2_extracted/M18KV2/train/rgb",
    annotations_file="/home/matson/M18K_dataset/M18KV2_extracted/M18KV2/train/annotations_coco.json",
    augmentations=augmentations,
    resize=(448,448)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)

val_dataset = MushroomCOCODataset(
    images_dir="/home/matson/M18K_dataset/M18KV2_extracted/M18KV2/valid/rgb",
    annotations_file="/home/matson/M18K_dataset/M18KV2_extracted/M18KV2/valid/annotations_coco.json",
    augmentations=None,
    resize=(448,448)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)

# visualize_samples(train_dataset, num_samples=5)
# visualize_samples(val_dataset, num_samples=3)

# -------- LOAD MODEL (Faster R-CNN) --------
weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

# Replace classifier head
num_classes = 2  # 1 class (mushroom) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Define optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0001)

# -------- NEW: LOAD CHECKPOINT --------
checkpoint_path = "best_fasterrcnn_mushroom_FULL.pth"
start_epoch = 1
best_val_loss = float('inf')

if os.path.exists(checkpoint_path):
    print(f"--- Loading Checkpoint: {checkpoint_path} ---")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    print(f"Resuming from Epoch {start_epoch}")

# -------- COSINE ANNEALING SCHEDULER --------
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10, eta_min=1e-6
)

# -------- TRAINING LOOP WITH BATCH SIZE 4 + OPTIONAL GRADIENT ACCUMULATION --------
def main():
    
    accum_steps = 8
    
    print("entering training")
    # num_epochs = 10
    global start_epoch, best_val_loss
    train_losses, val_losses = [], []
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, start_epoch + 10):

        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch}/{start_epoch + 9}]")
        
        for batch_idx, (images, targets) in enumerate(loop):

            def mem(label):
                torch.cuda.synchronize()
                alloc = torch.cuda.memory_allocated() / 1024**2
                peak  = torch.cuda.max_memory_allocated() / 1024**2
                print(f"  [Batch {batch_idx}] {label:<20} Alloc: {alloc:.1f} MB | Peak: {peak:.1f} MB")

            images = [img.to(device) for img in images]
            targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
            mem("after data→GPU")

            print(f"  [Batch {batch_idx}] boxes in image: {[len(t['boxes']) for t in targets]}")

            loss_dict = model(images, targets)
            mem("after forward")

            batch_loss = sum(loss for loss in loss_dict.values())
            print(f"  [Batch {batch_idx}] losses: { {k: f'{v.item():.4f}' for k,v in loss_dict.items()} }")

            (batch_loss / accum_steps).backward()
            mem("after backward")

            if (batch_idx + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                mem("after optimizer")

            epoch_loss += batch_loss.item()
            running_avg_loss = epoch_loss / (batch_idx + 1)

            loop.set_postfix(batch_loss=f"{batch_loss.item():.4f}", avg_loss=f"{running_avg_loss:.4f}")

            del images, targets, loss_dict, batch_loss
            torch.cuda.empty_cache()
            mem("after cleanup")

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        


        # -------- VALIDATION --------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                # Exclude masks for Faster R-CNN
                targets = [{k: v.to(device) for k, v in t.items() if k != "masks"} for t in targets]

                # Faster R-CNN returns dict of losses only in training mode
                # For evaluation, we can do a forward pass manually to get losses
                model.train()  # temporarily switch to training mode
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                model.eval()  # switch back to eval mode

        avg_val_loss = val_loss / len(val_loader)
        
        lr_scheduler.step()
        
        val_losses.append(avg_val_loss)

                # -------- PLOTTING AFTER EACH EPOCH --------
        plt.figure(figsize=(8,6))
        plt.plot(range(start_epoch, epoch+1), train_losses, label='Train Loss')
        plt.plot(range(start_epoch, epoch+1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"loss_curve_epoch_{epoch}.png")  # save each epoch's plot
        plt.close()

        print(f"\nEvaluating maP on validation set for epoch {epoch}...")
        evaluate_mAP(model, val_dataset, device, score_threshold=0.1)
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # # Step learning rate scheduler
        # lr_scheduler.step()

        # -------- SAVE CHECKPOINT --------
         # Save best model separately as a FULL checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                best_val_loss=best_val_loss,
                filename="best_fasterrcnn_mushroom_FULL.pth" # New name to be clear
            )
if __name__ == "__main__":
    main()

Epoch [28/37]:   0%|                                                                                                         | 0/3596 [00:00<?, ?it/s]  [Batch 0] after data→GPU       Alloc: 648.6 MB | Peak: 648.6 MB
  [Batch 0] boxes in image: [82]
  [Batch 0] after forward        Alloc: 1146.4 MB | Peak: 1601.9 MB
  [Batch 0] losses: {'loss_classifier': '0.0665', 'loss_box_reg': '0.1510', 'loss_objectness': '0.0036', 'loss_rpn_box_reg': '0.0155'}
  [Batch 0] after backward       Alloc: 809.6 MB | Peak: 1601.9 MB
Epoch [28/37]:   0%|                                                                     | 0/3596 [00:02<?, ?it/s, avg_loss=0.2366, batch_loss=0.2366]  [Batch 0] after cleanup        Alloc: 806.0 MB | Peak: 1601.9 MB
Epoch [28/37]:   0%|                                                           | 1/3596 [00:02<2:17:40,  2.30s/it, avg_loss=0.2366, batch_loss=0.2366]  [Batch 1] after data→GPU       Alloc: 809.6 MB | Peak: 1601.9 MB
  [Batch 1] boxes in image: [180]
  [Batch 1] after forward        Alloc: 1307.3 MB | Peak: 2321.6 MB
  [Batch 1] losses: {'loss_classifier': '0.0716', 'loss_box_reg': '0.1620', 'loss_objectness': '0.0230', 'loss_rpn_box_reg': '0.0238'}
  [Batch 1] after backward       Alloc: 809.6 MB | Peak: 2321.6 MB
Epoch [28/37]:   0%|                                                           | 1/3596 [00:03<2:17:40,  2.30s/it, avg_loss=0.2585, batch_loss=0.2804]  [Batch 1] after cleanup        Alloc: 806.0 MB | Peak: 2321.6 MB
Epoch [28/37]:   0%|                                                           | 2/3596 [00:03<1:38:35,  1.65s/it, avg_loss=0.2585, batch_loss=0.2804]  [Batch 2] after data→GPU       Alloc: 808.3 MB | Peak: 2321.6 MB
  [Batch 2] boxes in image: [173]
  [Batch 2] after forward        Alloc: 1295.8 MB | Peak: 2321.6 MB
  [Batch 2] losses: {'loss_classifier': '0.0764', 'loss_box_reg': '0.1029', 'loss_objectness': '0.0214', 'loss_rpn_box_reg': '0.0162'}
  [Batch 2] after backward       Alloc: 808.3 MB | Peak: 2321.6 MB
Epoch [28/37]:   0%|                                                           | 2/3596 [00:04<1:38:35,  1.65s/it, avg_loss=0.2447, batch_loss=0.2170]  [Batch 2] after cleanup        Alloc: 806.0 MB | Peak: 2321.6 MB
Epoch [28/37]:   0%|                                                           | 3/3596 [00:04<1:24:35,  1.41s/it, avg_loss=0.2447, batch_loss=0.2170]  [Batch 3] after data→GPU       Alloc: 808.3 MB | Peak: 2321.6 MB
  [Batch 3] boxes in image: [176]
  [Batch 3] after forward        Alloc: 1295.8 MB | Peak: 2321.6 MB
  [Batch 3] losses: {'loss_classifier': '0.0330', 'loss_box_reg': '0.0818', 'loss_objectness': '0.0167', 'loss_rpn_box_reg': '0.0159'}
  [Batch 3] after backward       Alloc: 808.3 MB | Peak: 2321.6 MB
Epoch [28/37]:   0%|                                                           | 3/3596 [00:05<1:24:35,  1.41s/it, avg_loss=0.2204, batch_loss=0.1475]  [Batch 3] after cleanup        Alloc: 806.0 MB | Peak: 2321.6 MB
Epoch [28/37]:   0%|                                                           | 4/3596 [00:05<1:18:01,  1.30s/it, avg_loss=0.2204, batch_loss=0.1475]  [Batch 4] after data→GPU       Alloc: 810.5 MB | Peak: 2321.6 MB
  [Batch 4] boxes in image: [127]
  [Batch 4] after forward        Alloc: 1307.4 MB | Peak: 2321.6 MB
  [Batch 4] losses: {'loss_classifier': '0.0868', 'loss_box_reg': '0.1584', 'loss_objectness': '0.0142', 'loss_rpn_box_reg': '0.0209'}
  [Batch 4] after backward       Alloc: 810.5 MB | Peak: 2321.6 MB
Epoch [28/37]:   0%|                                                           | 4/3596 [00:06<1:18:01,  1.30s/it, avg_loss=0.2324, batch_loss=0.2802]  [Batch 4] after cleanup        Alloc: 806.0 MB | Peak: 2321.6 MB
Epoch [28/37]:   0%|                                                           | 5/3596 [00:06<1:14:48,  1.25s/it, avg_loss=0.2324, batch_loss=0.2802]  [Batch 5] after data→GPU       Alloc: 810.5 MB | Peak: 2321.6 MB
  [Batch 5] boxes in image: [159]
  [Batch 5] after forward        Alloc: 1307.4 MB | Peak: 2321.6 MB
  [Batch 5] losses: {'loss_classifier': '0.0717', 'loss_box_reg': '0.1303', 'loss_objectness': '0.0083', 'loss_rpn_box_reg': '0.0252'}
  [Batch 5] after backward       Alloc: 810.5 MB | Peak: 2321.6 MB
Epoch [28/37]:   0%|                                                           | 5/3596 [00:07<1:14:48,  1.25s/it, avg_loss=0.2329, batch_loss=0.2355]  [Batch 5] after cleanup        Alloc: 806.0 MB | Peak: 2321.6 MB
Epoch [28/37]:   0%|                                                           | 6/3596 [00:08<1:13:05,  1.22s/it, avg_loss=0.2329, batch_loss=0.2355]  [Batch 6] after data→GPU       Alloc: 810.2 MB | Peak: 2321.6 MB
  [Batch 6] boxes in image: [207]
  [Batch 6] after forward        Alloc: 1307.4 MB | Peak: 2477.1 MB
  [Batch 6] losses: {'loss_classifier': '0.0627', 'loss_box_reg': '0.1579', 'loss_objectness': '0.0329', 'loss_rpn_box_reg': '0.0267'}
  [Batch 6] after backward       Alloc: 810.2 MB | Peak: 2477.1 MB
Epoch [28/37]:   0%|                                                           | 6/3596 [00:09<1:13:05,  1.22s/it, avg_loss=0.2396, batch_loss=0.2801]  [Batch 6] after cleanup        Alloc: 806.0 MB | Peak: 2477.1 MB
Epoch [28/37]:   0%|                                                           | 7/3596 [00:09<1:12:39,  1.21s/it, avg_loss=0.2396, batch_loss=0.2801]  [Batch 7] after data→GPU       Alloc: 810.2 MB | Peak: 2477.1 MB
  [Batch 7] boxes in image: [178]
  [Batch 7] after forward        Alloc: 1307.4 MB | Peak: 2477.1 MB
  [Batch 7] losses: {'loss_classifier': '0.0466', 'loss_box_reg': '0.1287', 'loss_objectness': '0.0126', 'loss_rpn_box_reg': '0.0309'}
  [Batch 7] after backward       Alloc: 810.2 MB | Peak: 2477.1 MB
  [Batch 7] after optimizer      Alloc: 810.2 MB | Peak: 2477.1 MB
Epoch [28/37]:   0%|                                                           | 7/3596 [00:10<1:12:39,  1.21s/it, avg_loss=0.2370, batch_loss=0.2189]  [Batch 7] after cleanup        Alloc: 806.0 MB | Peak: 2477.1 MB
Epoch [28/37]:   0%|▏                                                          | 8/3596 [00:10<1:12:31,  1.21s/it, avg_loss=0.2370, batch_loss=0.2189]  [Batch 8] after data→GPU       Alloc: 808.3 MB | Peak: 2477.1 MB
  [Batch 8] boxes in image: [162]
  [Batch 8] after forward        Alloc: 1295.8 MB | Peak: 2477.1 MB
  [Batch 8] losses: {'loss_classifier': '0.0573', 'loss_box_reg': '0.0946', 'loss_objectness': '0.0314', 'loss_rpn_box_reg': '0.0508'}
  [Batch 8] after backward       Alloc: 808.3 MB | Peak: 2477.1 MB
Epoch [28/37]:   0%|▏                                                          | 8/3596 [00:11<1:12:31,  1.21s/it, avg_loss=0.2367, batch_loss=0.2341]  [Batch 8] after cleanup        Alloc: 806.0 MB | Peak: 2477.1 MB
Epoch [28/37]:   0%|▏                                                          | 9/3596 [00:11<1:10:49,  1.18s/it, avg_loss=0.2367, batch_loss=0.2341]  [Batch 9] after data→GPU       Alloc: 808.3 MB | Peak: 2477.1 MB
  [Batch 9] boxes in image: [166]
  [Batch 9] after forward        Alloc: 1295.8 MB | Peak: 2477.1 MB
  [Batch 9] losses: {'loss_classifier': '0.0693', 'loss_box_reg': '0.1423', 'loss_objectness': '0.0598', 'loss_rpn_box_reg': '0.0882'}
  [Batch 9] after backward       Alloc: 808.3 MB | Peak: 2477.1 MB
Epoch [28/37]:   0%|▏                                                          | 9/3596 [00:12<1:10:49,  1.18s/it, avg_loss=0.2490, batch_loss=0.3596]  [Batch 9] after cleanup        Alloc: 806.0 MB | Peak: 2477.1 MB
Epoch [28/37]:   0%|▏                                                         | 10/3596 [00:12<1:09:47,  1.17s/it, avg_loss=0.2490, batch_loss=0.3596]  [Batch 10] after data→GPU       Alloc: 809.5 MB | Peak: 2477.1 MB
  [Batch 10] boxes in image: [170]
  [Batch 10] after forward        Alloc: 1346.1 MB | Peak: 2477.1 MB
  [Batch 10] losses: {'loss_classifier': '0.0784', 'loss_box_reg': '0.1522', 'loss_objectness': '0.0141', 'loss_rpn_box_reg': '0.0233'}
  [Batch 10] after backward       Alloc: 809.5 MB | Peak: 2477.1 MB
Epoch [28/37]:   0%|▏                                                         | 10/3596 [00:13<1:09:47,  1.17s/it, avg_loss=0.2507, batch_loss=0.2681]  [Batch 10] after cleanup        Alloc: 806.0 MB | Peak: 2477.1 MB
Epoch [28/37]:   0%|▏                                                         | 11/3596 [00:13<1:11:19,  1.19s/it, avg_loss=0.2507, batch_loss=0.2681]  [Batch 11] after data→GPU       Alloc: 809.5 MB | Peak: 2477.1 MB
  [Batch 11] boxes in image: [166]
  [Batch 11] after forward        Alloc: 1346.1 MB | Peak: 2477.1 MB
  [Batch 11] losses: {'loss_classifier': '0.0761', 'loss_box_reg': '0.1588', 'loss_objectness': '0.0259', 'loss_rpn_box_reg': '0.0198'}
Epoch [28/37]:   0%|▏  

  


