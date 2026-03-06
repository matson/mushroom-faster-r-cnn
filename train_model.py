# -------- CNN for Mushroom Detection --------

# disclaimer: many of the mushroom datasets I wanted to work with cost money to have more than 256 images...
# I stuck through whichever were available for free - I used M18K 

# Will be using MaskRCNN - pretrained on COCO 


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
'''

# -------- IMPORTS --------
import torch
from torch.utils.data import Dataset, DataLoader

# For visualizing predictions
import torchvision.transforms.functional as F
from matplotlib.patches import Rectangle

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms as T
import torchvision.transforms.functional as F
import random
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import visualize_samples
from utils import show_sample_image


# -------- SANITY CHECK --------

train_base = "~/M18K_dataset/M18KV2_extracted/M18KV2/train"
show_sample_image(train_base, subfolder="rgb", index=5)


# -------- CUSTOM DATASET CLASS --------

class MushroomCOCODataset(Dataset):
    def __init__(self, images_dir, annotations_file, transforms=None):
        self.images_dir = images_dir # images
        self.coco = COCO(annotations_file) # paired annotations 
        self.transforms = transforms
        
        # List of image IDs
        self.img_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_dir, img_info['file_name'])

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks = []

        for ann in anns:
            x, y, width, height = ann['bbox']

            # Skip invalid COCO boxes immediately
            if width <= 1 or height <= 1:
                continue

            boxes.append([x, y, x + width, y + height])
            labels.append(1)

            segm = ann['segmentation']
            rle = maskUtils.frPyObjects(segm, h, w)
            mask = maskUtils.decode(rle)
            if len(mask.shape) == 3:
                mask = np.any(mask, axis=2)

            masks.append(mask)

        # If no valid objects → skip image
        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks, dtype=np.uint8), dtype=torch.uint8)

        # ---- AUGMENTATIONS ----
        # * need to add data augmentations to reduce overfitting 
        if random.random() > 0.5:
            image = F.hflip(image)
            masks = torch.flip(masks, dims=[2])
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

        if random.random() > 0.5:
            image = F.vflip(image)
            masks = torch.flip(masks, dims=[1])
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]

        # ---- FINAL BOX VALIDATION ----
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        valid = (widths > 1) & (heights > 1)

        boxes = boxes[valid]
        labels = labels[valid]
        masks = masks[valid]

        # If augmentations killed all boxes → skip
        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id])
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target
    
    
# -------- DATALOADING --------

# training 
# CNN expects tensors, so we need to define transform 
transform = T.Compose([T.ToTensor()])

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)  # <- ensures targets is a list of dicts

train_dataset = MushroomCOCODataset(
    images_dir="/Users/madisonadams/M18K_dataset/M18KV2_extracted/M18KV2/train/rgb",
    annotations_file="/Users/madisonadams/M18K_dataset/M18KV2_extracted/M18KV2/train/annotations_coco.json",
    transforms=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)

# validation 
val_dataset = MushroomCOCODataset(
    images_dir="/Users/madisonadams/M18K_dataset/M18KV2_extracted/M18KV2/valid/rgb",
    annotations_file="/Users/madisonadams/M18K_dataset/M18KV2_extracted/M18KV2/valid/annotations_coco.json",
    transforms=transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn
)

# Visualize 5 samples from the training dataset
visualize_samples(train_dataset, num_samples=5)

# Or from validation dataset
visualize_samples(val_dataset, num_samples=3)


# -------- LOADING PRETRAINED MODEL --------

# # Load Mask R-CNN pretrained on COCO
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

# Replace box predictor (1 class + background)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Replace mask predictor
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)

# Load trained weights from the finished epoch
model.load_state_dict(torch.load("epoch1_latest.pth", map_location=device))
model.to(device)
print("Loaded trained model from epoch1_latest.pth")


# -------- HYPERPARAMS --------

params = [p for p in model.parameters() if p.requires_grad]

# Use SGD (common for detection models)
optimizer = torch.optim.SGD(
    params,
    lr=0.005,        # learning rate
    momentum=0.9,    # helps stabilize training
    weight_decay=0.0005  # regularization to prevent overfitting
)

# Optional: learning rate scheduler
# Reduce LR by 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# -------- TRAINING --------

num_epochs = 5  # you can increase later
best_val_loss = float('inf')
checkpoint_interval = 100  # save every 100 batches


for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]")
    for batch_idx, (images, targets) in enumerate(loop):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward + loss
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backprop
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Update tqdm
        epoch_loss += losses.item()
        loop.set_postfix(loss=epoch_loss / (batch_idx + 1))

        # --- Intermediate checkpoint ---
        if (batch_idx + 1) % checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch{epoch+1}_batch{batch_idx+1}.pth")

    # --- End-of-epoch checkpoint ---
    torch.save(model.state_dict(), f"epoch{epoch+1}_latest.pth")
    avg_train_loss = epoch_loss / len(train_loader)
    

    # -------- VALIDATION --------
    model.train()  # keep train mode so Mask R-CNN returns losses

    val_loss = 0

    with torch.no_grad():
        for images, targets in val_loader:

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            val_loss += losses.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Validation Loss: {avg_val_loss:.4f}")
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_maskrcnn_mushroom.pth")

    # Step scheduler per epoch
    lr_scheduler.step()

    
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
    for box in pred[0]['boxes']:
        x1, y1, x2, y2 = box.cpu().numpy()
        plt.gca().add_patch(Rectangle((x1,y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2))
    plt.show()

verify()