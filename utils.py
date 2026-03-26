import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from PIL import Image
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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


# utils.py (or wherever you keep evaluate_mAP)

from pycocotools.cocoeval import COCOeval
import torch
import numpy as np

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

            for box, score, label in zip(boxes, scores, labels):
                if score < score_threshold:
                    continue
                if label == 0:  # Skip background
                    continue

                x1, y1, x2, y2 = box
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)

                results.append({
                    "image_id": int(target['image_id'].item()),
                    "category_id": int(label),  # use predicted label
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
        if ann['category_id'] in [1, 2]:
            ann['category_id'] = 1

    # Replace categories with single class
    cocoGt.dataset['categories'] = [
        {"id": 1, "name": "mushroom"}
    ]
    # Rebuild index after modification
    cocoGt.createIndex()
    cocoDt = cocoGt.loadRes(results)

    # Run COCO evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval.params.useCats = 1  # ensure categories are used
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





if __name__ == "__main__":
    print("Running sanity check for mAP calculation...")

    import copy
    from pycocotools.cocoeval import COCOeval

    # Pick the first image from validation
    img, target = val_dataset[0]

    # --- Build fake predictions that exactly match GT ---
    results = []
    for box, label in zip(target['boxes'], target['labels']):
        x1, y1, x2, y2 = box.tolist()  # GT box coordinates in pixels
        results.append({
            "image_id": int(target['image_id'].item()),
            "category_id": 1,            # map all to single class
            "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO expects [x,y,width,height]
            "score": 1.0                 # perfect confidence
        })

    # --- Prepare COCO objects ---
    base_dataset = val_dataset
    cocoGt = copy.deepcopy(base_dataset.coco)

    # Map all categories to single class = 1
    for ann in cocoGt.dataset['annotations']:
        if ann['category_id'] in [1, 2]:  # adjust if you have multiple original classes
            ann['category_id'] = 1

    # Replace categories with single class
    cocoGt.dataset['categories'] = [{"id": 1, "name": "mushroom"}]
    cocoGt.createIndex()

    print("=== Debug Info BEFORE loadRes ===")
    print("GT first 5 annotations:", cocoGt.dataset['annotations'][:5])
    print("Pred first 5 results:", results[:5])
    
    gt_image_ids = [ann['image_id'] for ann in cocoGt.dataset['annotations']]
    pred_image_ids = [res['image_id'] for res in results]
    print("GT image_ids:", gt_image_ids[:5])
    print("Pred image_ids:", pred_image_ids[:5])
    
    gt_category_ids = [ann['category_id'] for ann in cocoGt.dataset['annotations']]
    pred_category_ids = [res['category_id'] for res in results]
    print("GT category_ids:", gt_category_ids[:5])
    print("Pred category_ids:", pred_category_ids[:5])
    
    gt_bboxes = [ann['bbox'] for ann in cocoGt.dataset['annotations']]
    pred_bboxes = [res['bbox'] for res in results]
    print("GT bboxes:", gt_bboxes[:5])
    print("Pred bboxes:", pred_bboxes[:5])

    # Load the results
    cocoDt = cocoGt.loadRes(results)

    # --- Run COCO evaluation ---
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # AP50 specifically
    print("Sanity check AP50:", cocoEval.stats[1])  # index 1 = AP at IoU=0.50

    # Stop execution after sanity check
    raise RuntimeError("Sanity check done")


