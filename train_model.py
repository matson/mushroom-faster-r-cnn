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
pip install albumentations
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
import albumentations as A

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import visualize_samples
from utils import show_sample_image

'''
handle COCO style datasets - used for object detection 
masks tell model the precise outline of each mushroom 
Bounding boxes - rectangle around mushroom 
segmentation - exact pixels of mushroom 
image_id - links annotation to correct image 

Resize images - will make everything smoother 
Most images are JPEG - already RGB
Bottleneck is disk space 


'''


# -------- SANITY CHECK --------

train_base = "~/M18K_dataset/M18KV2_extracted/M18KV2/train"
show_sample_image(train_base, subfolder="rgb", index=5)

# -------- TRANSFORMS DEF --------
augmentations = A.Compose([
    A.Affine(
        scale=(0.8, 1.2),
        rotate=(-360, 360),
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
    def __init__(self, images_dir, annotations_file, augmentations=None, resize=(512, 512)):
        # loads paths - images, COCO annot, masks, transformations 
        self.images_dir = images_dir 
        self.coco = COCO(annotations_file) 
        self.augmentations = augmentations
        self.resize = resize  # (width, height)
        self.img_ids = list(self.coco.imgs.keys()) # List of image IDs

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB") # convert to rgb
        w_original, h_original = image.size

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
            labels.append(1) # mushroom class 

            segm = ann['segmentation']
            rle = maskUtils.frPyObjects(segm, h_original, w_original)
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

        # ---- RESIZE IMAGE AND ANNOTATIONS ----
        w_new, h_new = self.resize
        scale_x = w_new / w_original
        scale_y = h_new / h_original

        # Resize image
        image = image.resize((w_new, h_new), resample=Image.BILINEAR)

        # Scale boxes
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y

        # Resize masks
        resized_masks = []
        for mask in masks:
            mask_pil = Image.fromarray(mask.numpy())
            mask_resized = mask_pil.resize((w_new, h_new), resample=Image.NEAREST)
            resized_masks.append(torch.as_tensor(np.array(mask_resized), dtype=torch.uint8))
        masks = torch.stack(resized_masks)

        # ---- AUGMENTATIONS ----
        # * need to add data augmentations to reduce overfitting 
        if self.augmentations:
            # Convert image/masks to numpy for Albumentations
            transformed = self.augmentations(
                image=np.array(image),
                masks=[m.numpy() for m in masks],
                bboxes=boxes.numpy(),
                labels=labels.numpy()
            )
            image = transformed['image']
            masks = torch.as_tensor(transformed['masks'], dtype=torch.uint8)
            boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.as_tensor(transformed['labels'], dtype=torch.int64)

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

        return image, target
    
    
# -------- DATALOADING --------

transform = T.Compose([T.ToTensor()]) # CNN expects tensors, so we need to define transform 

# --- SETTINGS ---
num_epochs = 5
best_val_loss = float('inf')
checkpoint_interval = 100  # save every N batches
batch_size = 4             # increased from 2
accum_steps = 1            # set >1 if GPU memory is limited

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)  # ensures targets is a list of dicts

train_dataset = MushroomCOCODataset(
    images_dir="/Users/madisonadams/M18K_dataset/M18KV2_extracted/M18KV2/train/rgb",
    annotations_file="/Users/madisonadams/M18K_dataset/M18KV2_extracted/M18KV2/train/annotations_coco.json",
    transforms=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
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
    batch_size=1,  # keep validation batch small
    shuffle=False,
    collate_fn=collate_fn
)

visualize_samples(train_dataset, num_samples=5) # Visualize 5 samples from the training dataset
visualize_samples(val_dataset, num_samples=3) # Or from validation dataset

# -------- LOADING PRETRAINED MODEL --------

# # Load Mask R-CNN pretrained on COCO ***
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


# -------- TRAINING LOOP WITH BATCH SIZE 4 + OPTIONAL GRADIENT ACCUMULATION --------
for epoch in range(num_epochs):

    train_losses = [] # for plotting 
    val_losses = []

    model.train()
    epoch_loss = 0
    optimizer.zero_grad()  # initialize for accumulation

    loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]")
    for batch_idx, (images, targets) in enumerate(loop):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward + loss
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses = losses / accum_steps  # normalize for accumulation
        losses.backward()

        # Gradient step every accum_steps
        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Update tqdm
        epoch_loss += losses.item() * accum_steps  # multiply back for reporting
        loop.set_postfix(loss=epoch_loss / (batch_idx + 1))

        # --- Intermediate checkpoint ---
        if (batch_idx + 1) % checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch{epoch+1}_batch{batch_idx+1}.pth")

    # --- End-of-epoch checkpoint ---
    torch.save(model.state_dict(), f"epoch{epoch+1}_latest.pth")
    avg_train_loss = epoch_loss / len(train_loader)

    # -------- VALIDATION --------
    model.train()  # Mask R-CNN returns losses in train mode
    val_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            val_loss += sum(loss for loss in loss_dict.values()).item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    train_losses.append(avg_train_loss) # end of each epoch...
    val_losses.append(avg_val_loss)

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_maskrcnn_mushroom.pth")

    # Step scheduler per epoch
    lr_scheduler.step()

# -------- PLOTTING --------

plt.figure(figsize=(8,6))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

    
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
