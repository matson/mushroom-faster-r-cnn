import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import albumentations as A


class MushroomCOCODataset(Dataset):
    def __init__(self, images_dir, annotations_file, augmentations=None, resize=(448, 448)):
        self.images_dir = images_dir
        self.coco = COCO(annotations_file)
        self.augmentations = augmentations
        self.resize = resize

        self.img_ids = [
            img_id for img_id in self.coco.imgs.keys()
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        w_original, h_original = image.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            if ann['category_id'] == 0:
                continue
            x, y, width, height = ann['bbox']
            if width <= 1 or height <= 1:
                continue
            boxes.append([x, y, x + width, y + height])
            labels.append(1)

        if len(boxes) == 0:
            raise ValueError(f"Image {img_id} has no valid mushrooms.")

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        w_new, h_new = self.resize
        scale_x = w_new / w_original
        scale_y = h_new / h_original
        image = image.resize((w_new, h_new), resample=Image.BILINEAR)
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        if self.augmentations:
            transformed = self.augmentations(
                image=np.array(image),
                bboxes=boxes.numpy().tolist(),
                labels=labels.numpy().tolist()
            )
            image = transformed['image']
            boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.as_tensor([int(l) for l in transformed['labels']], dtype=torch.int64)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        valid = (widths > 1) & (heights > 1)
        boxes = boxes[valid]
        labels = labels[valid]

        image = torch.as_tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        return image, target

def visualize_samples(dataset, num_samples=3):
    """
    Visualize a few samples from a COCO-style dataset with bounding boxes and masks.

    Args:
        dataset: A PyTorch Dataset object (returns (image, target) tuples).
        num_samples (int): Number of samples to visualize.
    """
    samples = [dataset[i] for i in range(num_samples)]

    plt.figure(figsize=(12, 4 * num_samples))

    for i, (img, target) in enumerate(samples):
        img_np = img.permute(1, 2, 0).numpy()  # [H, W, C] for plotting

        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(img_np)
        plt.axis('off')

        # Plot bounding boxes
        for box in target['boxes']:
            x1, y1, x2, y2 = box.numpy()
            plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                          fill=False, color='red', linewidth=2))

        # Plot masks
        for mask in target['masks']:
            mask_np = mask.numpy()
            plt.imshow(mask_np, alpha=0.4)

    plt.tight_layout()
    plt.show()


def show_sample_image(base_path, subfolder="rgb", index=0):
    """
    Display a sample image from a dataset folder.

    Args:
        base_path (str): Path to the dataset (e.g., "~/M18K_dataset/M18KV2_extracted/M18KV2/train")
        subfolder (str): Subfolder containing images (default "rgb")
        index (int): Index of the image to display (default 0)
    """
    images_path = os.path.join(os.path.expanduser(base_path), subfolder)
    img_files = sorted(os.listdir(images_path))
    
    if index >= len(img_files):
        raise IndexError(f"Index {index} is out of range. There are only {len(img_files)} images.")
    
    img_file = img_files[index]
    img = Image.open(os.path.join(images_path, img_file)).convert("RGB")
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Sample Image: {img_file}")
    plt.axis('off')
    plt.savefig("sample_image_check.png")
    plt.close()


def evaluate_mAP(model, val_dataset, device, score_threshold=0.1):
    """
    Compute COCO-style mAP and AR for a validation dataset.

    Args:
        model: torch model (Faster R-CNN)
        val_dataset: Dataset or Subset
        device: 'cpu' or 'cuda'
        score_threshold: float, minimum prediction score to include
    """

    # Handle Subset vs full dataset
    if isinstance(val_dataset, torch.utils.data.Subset):
        base_dataset = val_dataset.dataset
    else:
        base_dataset = val_dataset

    model.eval()
    results = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for idx in range(len(val_dataset)):
            img, target = val_dataset[idx]
            img = img.to(device)

            pred = model([img])[0]

            # Get boxes, scores, and labels
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()

            # Scale predicted boxes from 256x256 back to original image coords
            img_id = int(target['image_id'].item())
            img_info = base_dataset.coco.loadImgs(img_id)[0]
            w_orig, h_orig = img_info['width'], img_info['height']
            w_new, h_new = base_dataset.resize
            scale_x = w_orig / w_new
            scale_y = h_orig / h_new

            for box, score, label in zip(boxes, scores, labels):
                if score < score_threshold:
                    continue
                if label == 0:  # Skip background
                    continue

                x1, y1, x2, y2 = box
                x1 *= scale_x;  x2 *= scale_x
                y1 *= scale_y;  y2 *= scale_y
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)

                results.append({
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(width), float(height)],
                    "score": float(score)
                })

    if len(results) == 0:
        print("No predictions above threshold!")
        return

    # Load ground truth COCO object
    # safe copy - no real data changing 
    import copy
    cocoGt = copy.deepcopy(base_dataset.coco)

    # -------- FIX CATEGORY IDS (MERGE INTO SINGLE CLASS) --------
    for ann in cocoGt.dataset['annotations']:
        ann['category_id'] = 1  # remap all to mushroom

    # Replace categories with single class
    cocoGt.dataset['categories'] = [
        {"id": 1, "name": "mushroom"}
    ]
    # Rebuild index after modification
    cocoGt.createIndex()
    cocoDt = cocoGt.loadRes(results)

    # Set maxDets to cover densest image in dataset
    from collections import Counter
    counts = Counter(ann['image_id'] for ann in cocoGt.dataset['annotations'])
    max_dets = max(counts.values())

    # Run COCO evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval.params.useCats = 1  # ensure categories are used
    cocoEval.params.maxDets = [1, 10, max_dets]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print(f"AP50 (IoU=0.5): {cocoEval.stats[1]:.4f}")  # convenient quick metric


# -------- SAVE CHECKPOINT --------
def save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, filename="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

# -------- LOAD CHECKPOINT --------
def load_checkpoint(filename, model, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and scheduler:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["best_val_loss"]
    print(f"Checkpoint loaded: {filename} (epoch {epoch})")
    return epoch, best_val_loss
