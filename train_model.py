
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
        assert len(boxes) == len(labels), f"Boxes ({len(boxes)}) and labels ({len(labels)}) mismatch!"

        # 🚨 Stop here for inspection
        raise RuntimeError(f"Inspecting sample {idx}: boxes={boxes}, labels={labels}, image.shape={image.shape}")

        return image, target




from torch.utils.data import DataLoader

# -------- DATALOADING --------
num_epochs = 5
best_val_loss = float('inf')
checkpoint_interval = 100
batch_size = 1     # smaller to avoid memory issues on Mac
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

for i in range(3):  # first 3 images
    img, target = train_dataset[i]
    print(f"Image {i}:")
    print(f"Boxes: {target['boxes'].numpy()}")
    print(f"Labels: {target['labels'].numpy()}")
    print("----")

from torch.utils.data import DataLoader

# Use a small DataLoader to avoid memory issues
temp_loader = DataLoader(
    train_dataset, 
    batch_size=1, 
    shuffle=False, 
    collate_fn=lambda x: x  # simple collate
)

print("Printing first 10 samples with COCO category_id -> model label mapping:\n")
for i, data in enumerate(temp_loader):
    img, target = data[0]  # unpack the batch
    boxes = target['boxes']
    labels = target['labels']
    coco_ids = []

    # Get COCO category_ids from the dataset directly
    img_id = target['image_id'].item()
    ann_ids = train_dataset.coco.getAnnIds(imgIds=img_id)
    anns = train_dataset.coco.loadAnns(ann_ids)
    for ann in anns:
        coco_ids.append(ann['category_id'])

    print(f"Sample {i+1}:")
    print(f"Image ID: {img_id}")
    print(f"COCO category_ids: {coco_ids}")
    print(f"Model labels: {labels.tolist()}")
    print(f"Boxes: {boxes.tolist()}\n")

    if i >= 9:  # only first 10
        break


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

visualize_samples(train_dataset, num_samples=5)
visualize_samples(val_dataset, num_samples=3)

# -------- LOAD MODEL (Faster R-CNN) --------
weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

# Replace classifier head
num_classes = 2  # 1 class (mushroom) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Load the best model weights
model.load_state_dict(torch.load("best_maskrcnn_mushroom.pth", map_location=device))
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# -------- TRAINING LOOP WITH BATCH SIZE 4 + OPTIONAL GRADIENT ACCUMULATION --------
def main():
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

            optimizer.step()
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
    plt.plot(range(6, 11), train_losses, label='Train Loss')
    plt.plot(range(6, 11), val_losses, label='Validation Loss')
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

# 0.005
# batch 1

'''
data 

Traceback (most recent call last):
  File "/home/matson/mushroom-mask-rcnn/train_model.py", line 197, in <module>
    img, target = train_dataset[i]
  File "/home/matson/mushroom-mask-rcnn/train_model.py", line 169, in __getitem__
    raise RuntimeError(f"Inspecting sample {idx}: boxes={boxes}, labels={labels}, image.shape={image.shape}")
RuntimeError: Inspecting sample 0: boxes=tensor([[ 97.8560,  42.2429, 124.1564,  87.1023],
        [119.6431, 227.1516, 143.8937, 267.4338],
        [253.7077, 187.0387, 273.6561, 219.7220],
        [239.0116, 249.8757, 259.2544, 281.8034],
        [ 82.6899, 111.1242, 110.6079, 160.1165],
        [181.1854,  80.9569, 211.3099, 132.5708],
        [ 47.3604, 221.4843,  68.5783, 251.9450],
        [139.0799, 187.2473, 159.5697, 226.3084],
        [ 16.4806, 130.6809,  45.4021, 173.4727],
        [ 72.3884,  25.4786,  92.4286,  58.9619],
        [ 26.5367, 201.1255,  56.6628, 244.4503],
        [ 65.1428, 125.6504,  88.7318, 163.4883],
        [ 71.8467, 224.7595,  97.6421, 265.2190],
        [144.9229,  62.5439, 165.1927,  98.0272],
        [241.2652, 194.2796, 263.7145, 228.8289],
        [ 96.8042, 164.4587, 121.9095, 207.2074],
        [220.0850, 112.3426, 239.5459, 145.7595],
        [166.0263,   8.9987, 189.4791,  37.3475],
        [127.6023,  94.5846, 159.7955, 147.6201],
        [ 87.0278, 277.3606, 108.3185, 311.7769],
        [224.9383, 173.1420, 246.8906, 210.0026],
        [243.6990, 108.2396, 264.1081, 136.6338],
        [ 41.3892,  86.9794,  69.7678, 131.6825],
        [167.8728, 172.1682, 188.6476, 212.0514],
        [ 87.4106, 245.0729, 107.2024, 279.7119],
        [239.6968, 223.1004, 263.7179, 261.3826],
        [206.4401, 210.1555, 224.5303, 246.5728],
        [102.4397, 273.2997, 126.2771, 309.9819],
        [195.5816, 153.0880, 213.0561, 187.4610],
        [123.5854, 138.1587, 154.7023, 193.4390],
        [ 65.8210, 241.1550,  84.8403, 275.7053],
        [219.2673, 210.6718, 241.9540, 253.9323],
        [ 78.9952,  49.9038, 107.8044, 103.3404],
        [ 37.2036, 248.8402,  76.9454, 307.8735],
        [133.1420,  72.5394, 151.5646, 101.8898],
        [204.3826, 247.4670, 232.5492, 295.3036],
        [ 88.5436, 183.3871, 107.5819, 214.7818],
        [163.4915, 145.8128, 184.1300, 176.2070],
        [184.8981, 247.0371, 206.8504, 283.8977],
        [ 64.2483,  89.6718,  80.4265, 122.7120],
        [ 41.6652, 189.1500,  60.7589, 219.3669],
        [ 17.7103,  93.6386,  46.9626, 137.6526],
        [256.0109, 213.8384, 277.0989, 249.8104],
        [253.5293, 259.0346, 271.4454, 292.2742],
        [161.3979, 231.6302, 183.0179, 275.5577],
        [159.0023, 104.6143, 184.3926, 148.1851],
        [177.8819,  43.8238, 205.2221,  84.4605],
        [134.6798, 213.4897, 156.1731, 246.3503],
        [249.3627, 162.7719, 269.3301, 192.2996],
        [237.5381, 263.2064, 252.5671, 294.5357],
        [ 40.0124, 124.1335,  71.9112, 177.9247],
        [206.2712,  23.4118, 236.2390,  76.9813],
        [135.9339,  40.2760, 157.9875,  76.3588],
        [ 62.1483, 158.4057,  94.2593, 209.0634],
        [ 98.9761, 206.3220, 115.5784, 233.0952],
        [144.1780, 164.5184, 170.6905, 206.2444],
        [141.4243, 240.0013, 161.9885, 274.7289],
        [123.7115,  51.9369, 140.6715,  83.4878],
        [200.4952, 236.7524, 217.9522, 265.9921],
        [220.1765,  63.7005, 239.0407,  91.9174],
        [195.5142, 271.5942, 214.3325, 299.4111],
        [ 48.8661,  30.2708,  67.9598,  60.4877],
        [228.7869,  43.0824, 249.5632,  74.6766],
        [147.5208, 138.1631, 166.1270, 169.1134],
        [ 86.6174, 207.7403, 103.5870, 237.7134],
        [ 28.5593, 152.5942,  55.7887, 195.5866],
        [236.4021,  85.3944, 258.6773, 116.7658],
        [123.5040,  29.4053, 142.4600,  58.4222],
        [205.1119, 181.8666, 219.8007, 213.5516],
        [ 98.7548, 260.6971, 114.8332, 286.2259],
        [182.5503, 161.5743, 208.4075, 215.8563],
        [126.5761, 279.4391, 145.9738, 307.3225],
        [204.4971,  72.0949, 222.8548, 104.2010],
        [ 54.3363, 290.1517,  80.5115, 315.6551],
        [ 29.9685, 241.5462,  56.9415, 278.9829],
        [123.9085, 182.9095, 144.8574, 225.9705],
        [108.5364, 221.4354, 125.2764, 249.4085],
        [ 85.2385, 154.4545, 105.5731, 187.1821],
        [165.4647,  64.1283, 188.8116, 118.1222],
        [  4.9272,  52.5909,  31.8781, 109.7615],
        [ 32.4682,  44.8014,  63.1546,  91.3484],
        [ 56.6071, 191.9090,  72.0318, 221.7048],
        [ 89.3800,  32.4602, 104.2823,  52.7226],
        [148.5588,  93.0090, 163.9836, 122.8048],
        [237.8903, 144.8575, 255.5595, 170.9636],
        [ 77.7065, 262.9353,  96.2019, 296.2414],
        [168.5730, 266.6979, 189.3508, 290.0030],
        [176.7627,  29.8439, 194.7722,  55.5944],
        [  8.4535,  26.1291,  40.7147,  59.8308],
        [ 74.8880,  84.4109,  98.6147, 123.4487],
        [241.9972,  61.4401, 259.8390,  99.0131],
        [176.3223, 228.6573, 198.2367, 271.8292],
        [267.2745, 268.4151, 280.3741, 291.2337],
        [ 57.0003,  49.4739,  82.1975,  93.0225],
        [153.1244, 219.6713, 168.1645, 241.1337],
        [233.0318,  10.3736, 253.0341,  50.1682],
        [154.9070, 276.6945, 180.3462, 304.0871],
        [229.7148,  99.7444, 241.8219, 118.8966],
        [193.8333,   6.4502, 211.2270,  30.1564],
        [ 68.3754, 199.7371,  91.9659, 229.2859],
        [184.7724, 128.6639, 203.4973, 163.9699],
        [214.6885,   3.6801, 234.6305,  21.3631],
        [ 16.1737, 151.2990,  35.6141, 196.1606],
        [164.7312, 206.3651, 182.6663, 236.4490],
        [ 22.3699,  56.3515,  47.2631, 102.2336],
        [159.4198,  34.0378, 175.4254,  55.6110],
        [ 95.2383,  77.0414, 125.4546, 129.4553],
        [101.8834, 146.2373, 119.9387, 172.3878],
        [100.9867,  19.6193, 114.1322,  42.8379],
        [166.9334,  28.1287, 181.7424,  55.8803],
        [158.5084,  91.2782, 170.9922, 112.0525],
        [148.2053, 268.6902, 164.1935, 285.1299],
        [179.1586, 205.6149, 199.7971, 236.0090],
        [ 28.4534, 180.0122,  44.4400, 204.7411],
        [117.4522, 190.6158, 128.7490, 215.9906],
        [201.5023,  56.8042, 214.0131,  81.1342],
        [113.8888, 114.8194, 129.2218, 143.8152],
        [ 69.6310, 274.9308,  82.3902, 298.1051],
        [ 34.7616,  24.6454,  55.2924,  40.8171],
        [115.0383, 165.5734, 131.6406, 192.3465],
        [154.7641, 203.2726, 170.0337, 226.7350],
        [120.6506,  96.9558, 137.1595, 131.2181],
        [ 25.7123, 236.6964,  38.2040, 264.1820],
        [178.5312, 115.5505, 193.6616, 146.1020],
        [109.9613,  18.6464, 130.1772,  47.0185],
        [177.6553, 280.6567, 201.5418, 301.1607],
        [224.5573, 247.0260, 241.5268, 276.9991],
        [ 30.2204, 274.1630,  44.6416, 310.1593],
        [ 12.9604, 123.8312,  25.9302, 152.1611],
        [201.2854,  36.4710, 210.8631,  58.4907],
        [ 64.0644,  38.4968,  80.0700,  60.0700],
        [169.5594,  49.7939, 179.0184,  67.4580],
        [ 58.2743,  21.1208,  82.7496,  40.1134],
        [153.9559,  45.0560, 163.9389,  63.9644],
        [190.2246,  30.0467, 202.6356,  46.8653],
        [149.5365, 280.9327, 160.6591, 303.1297],
        [139.7801,  27.0972, 150.5544,  42.9387],
        [ 28.8891, 263.1615,  36.2604, 282.5596],
        [ 22.7380, 209.3848,  32.7558, 238.5601],
        [197.7124, 217.9737, 211.4372, 241.2588],
        [132.2047,  19.0065, 143.7325,  38.0922],
        [157.0104, 240.4922, 168.1678, 272.9561],
        [119.3303,  75.5333, 131.9424,  99.0855],
        [ 34.2804, 296.1672,  59.4173, 317.6041],
        [199.5220,  24.1165, 211.8412,  40.1352],
        [253.5333, 135.0393, 266.9082, 160.2579],
        [192.4971, 227.5482, 201.9372, 248.3679],
        [  2.8156,  35.0776,  18.2783,  58.5621],
        [ 77.2508, 291.7106,  91.3253, 313.0622],
        [222.9715,  15.1692, 237.6617,  38.5651],
        [212.1845, 274.0030, 242.1395, 297.5717],
        [115.1901, 139.6634, 132.8751, 179.1921],
        [205.7335, 100.8913, 214.5388, 122.8224],
        [246.3545,  18.6406, 254.3305,  49.9499],
        [136.5034, 249.0860, 148.8210, 273.3938],
        [ 69.8814,  58.6084,  86.6199,  94.8707],
        [ 11.1701, 108.9405,  17.1802, 129.7613],
        [193.7718,  42.3801, 205.2806,  64.6214],
        [259.9966, 242.2592, 275.6334, 268.9215],
        [188.8285,  21.4240, 199.4841,  32.9098],
        [220.2949, 274.0030, 242.1395, 296.6410],
        [112.8628,  42.8445, 126.9373,  64.1961],
        [272.9963, 255.8380, 280.4041, 277.2139],
        [255.7838,  97.4397, 261.3997, 111.5049],
        [126.8886, 128.0436, 135.9804, 142.5077],
        [ 21.4068, 198.3168,  24.9539, 210.9604],
        [233.7287,  73.5685, 243.7670,  91.2991],
        [259.9966, 242.2592, 274.5317, 259.3216],
        [ 72.3446, 116.2943,  83.4403, 134.9356],
        [161.4080,  74.4119, 169.8272,  96.2987],
        [113.3863, 249.7032, 134.4555, 288.8308],
        [ 85.6088, 141.3421,  98.3870, 161.3608],
        [242.2319, 137.3241, 252.5013, 148.7655],
        [ 34.3058, 310.6491,  37.7897, 317.7592],
        [201.9384, 188.8195, 211.3595, 212.7948],
        [128.2211,  81.8014, 136.5769,  98.1547],
        [250.3022,  53.0402, 255.7994,  62.7498],
        [158.3040,  51.7030, 172.7014,  67.5655],
        [245.0961,  11.0406, 250.8877,  19.9946],
        [  2.0811,  29.3203,  10.1994,  36.9623]]), labels=tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), image.shape=torch.Size([3, 319, 283])

'''
