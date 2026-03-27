
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
    def __init__(self, images_dir, annotations_file, augmentations=None, resize=(256, 256)):
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
    resize=(256,256)
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
    resize=(256,256)
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

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# -------- TRAINING LOOP WITH BATCH SIZE 4 + OPTIONAL GRADIENT ACCUMULATION --------
def main():
    print("entering training")
    # num_epochs = 10
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(1,11):

        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch}/{10}]")
        
        for batch_idx, (images, targets) in enumerate(loop):

            images = [img.to(device) for img in images]
            targets = [{k:v.to(device) for k,v in t.items()} for t in targets]

            # -------- GPU MEMORY LOGGING --------
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # sync to get accurate memory usage
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                peak = torch.cuda.max_memory_allocated() / 1024**2
                print(f"[Batch {batch_idx}] Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB | Peak: {peak:.1f} MB")
                print(f"[Batch {batch_idx}] Objects in batch: {[len(t['boxes']) for t in targets]}")

            loss_dict = model(images, targets)
            batch_loss = sum(loss for loss in loss_dict.values())  # <-- this batch's loss
            batch_loss.backward()

            optimizer.step() # update model weights 
            optimizer.zero_grad()

            epoch_loss += batch_loss.item()  # accumulate epoch loss
            running_avg_loss = epoch_loss / (batch_idx + 1)     # average so far

            loop.set_postfix(batch_loss=f"{batch_loss.item():.4f}", avg_loss=f"{running_avg_loss:.4f}")


            del images, targets, loss_dict, batch_loss
            torch.cuda.empty_cache()

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
        val_losses.append(avg_val_loss)

                # -------- PLOTTING AFTER EACH EPOCH --------
        plt.figure(figsize=(8,6))
        plt.plot(range(1, epoch+1), train_losses, label='Train Loss')
        plt.plot(range(1, epoch+1), val_losses, label='Validation Loss')
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

        # Step learning rate scheduler
        lr_scheduler.step()

        # -------- SAVE CHECKPOINT --------
        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            best_val_loss=best_val_loss,
            filename=f"checkpoint_epoch_{epoch}.pth"
        )

        # Save best model separately
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_fasterrcnn_mushroom.pth")

    # -------- PLOTTING --------

    plt.figure(figsize=(8,6))
    plt.plot(range(1, 11), train_losses, label='Train Loss')
    plt.plot(range(1, 11), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.close()
if __name__ == "__main__":
    main()

  
# -------- FINAL IMAGE VERIFICATION --------
def verify(): 
    model.eval()
    with torch.no_grad():
        # pick one validation image
        img, target = val_dataset[0]
        pred = model([img.to(device)])

    # visualize
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    img_np = img.permute(1,2,0).numpy()
    plt.imshow(img_np)
    for box, score in zip(pred[0]['boxes'], pred[0]['scores']):
        if score > 0.5:  # filter weak predictions
            x1, y1, x2, y2 = box.cpu().numpy()
            plt.gca().add_patch(
                Rectangle((x1, y1), x2 - x1, y2 - y1,
                        fill=False, color='red', linewidth=2)
            )
    plt.savefig("prediction_example.png")
    plt.close()
verify()

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import copy
import numpy as np
import torch

cocoGt = copy.deepcopy(val_dataset.coco)
predictions = []

for img_idx in range(len(val_dataset)):
    img, target = val_dataset[img_idx]
    
    gt_boxes = target['boxes'].clone()  # already resized
    gt_labels = target['labels']
    gt_scores = torch.ones(len(gt_boxes))  # dummy perfect confidence
    
    # Scale back to original image size
    img_info = cocoGt.loadImgs(int(target['image_id']))[0]
    w_orig, h_orig = img_info['width'], img_info['height']
    w_new, h_new = val_dataset.resize
    scale_x = w_orig / w_new
    scale_y = h_orig / h_new
    gt_boxes[:, [0, 2]] *= scale_x
    gt_boxes[:, [1, 3]] *= scale_y

    # Convert to COCO [x, y, w, h]
    for box, label, score in zip(gt_boxes, gt_labels, gt_scores):
        x1, y1, x2, y2 = box
        predictions.append({
            "image_id": int(target['image_id']),
            "category_id": int(label),
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(score)
        })

# Evaluate
coco_dt = cocoGt.loadRes(predictions)
coco_eval = COCOeval(cocoGt, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
