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
    # Sanity check for mAP with resized dataset
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np

    """
    dataset: your MushroomCOCODataset
    predictions: list of model outputs for each image
                 each element = dict with keys: 'boxes', 'scores', 'labels'
    """
    coco_gt = dataset.coco
    scaled_predictions = []

    print("Running sanity check for mAP calculation...")

    for pred, target in zip(predictions, dataset):
        # Extract info
        boxes = pred['boxes'].cpu().numpy()       # [N, 4]
        scores = pred['scores'].cpu().numpy()     # [N]
        labels = pred['labels'].cpu().numpy()     # [N]
        img_id = int(target['image_id'])
        img_info = coco_gt.loadImgs(img_id)[0]
        w_orig, h_orig = img_info['width'], img_info['height']

        # Dataset resize
        w_new, h_new = dataset.resize
        scale_x = w_orig / w_new
        scale_y = h_orig / h_new

        # Debug print
        print(f"\n=== Image {img_id} ===")
        print(f"Original size: ({w_orig}, {h_orig}), Resized: ({w_new}, {h_new})")
        print(f"Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")
        print("Pred boxes before scaling:")
        print(boxes[:5])

        # Rescale boxes back to original image size
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        print("Pred boxes after scaling:")
        print(boxes[:5])

        # Convert to COCO format [x, y, width, height]
        coco_boxes = []
        for b in boxes:
            x, y, x2, y2 = b
            coco_boxes.append([x, y, x2 - x, y2 - y])

        for b, s, l in zip(coco_boxes, scores, labels):
            scaled_predictions.append({
                "image_id": img_id,
                "category_id": int(l),
                "bbox": [float(coord) for coord in b],
                "score": float(s)
            })

    print("\nLoading and preparing results...")
    coco_dt = coco_gt.loadRes(scaled_predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

    print("Running per image evaluation...")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

import copy
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# --- Use your validation dataset ---
base_dataset = val_dataset  # MushroomCOCODataset instance
cocoGt = copy.deepcopy(base_dataset.coco)

# --- Fix category IDs ---
for ann in cocoGt.dataset['annotations']:
    if ann['category_id'] in [1, 2]:
        ann['category_id'] = 1
cocoGt.dataset['categories'] = [{"id": 1, "name": "mushroom"}]
cocoGt.createIndex()

# --- Prepare GT boxes as "predictions" for first 3 images ---
predictions = []
for img_idx in range(min(3, len(base_dataset))):
    image, target = base_dataset[img_idx]
    
    boxes = target['boxes'].cpu().numpy()
    labels = target['labels'].cpu().numpy()
    scores = np.ones(len(boxes))  # dummy score for perfect match

    # Convert to COCO format [x,y,w,h]
    coco_boxes = []
    for b in boxes:
        x1, y1, x2, y2 = b
        coco_boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

    # Add to predictions
    for b, s, l in zip(coco_boxes, scores, labels):
        predictions.append({
            "image_id": int(target['image_id'].item()),  # important: convert tensor to int
            "category_id": int(l),
            "bbox": b,
            "score": float(s)
        })

# --- Print first 3 predictions for sanity check ---
print("\n=== First 3 images predictions ===")
for pred in predictions[:15]:  # first 3 images × ~5 boxes each
    print(f"image_id={pred['image_id']}, category_id={pred['category_id']}, bbox={pred['bbox']}, score={pred['score']}")

# --- Run COCOeval ---
coco_dt = cocoGt.loadRes(predictions)
coco_eval = COCOeval(cocoGt, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()
print("\n=== Sanity mAP summary ===")
coco_eval.summarize()

ndex created!
creating index...
index created!

=== First 3 images predictions ===
image_id=0, category_id=1, bbox=[5.0, 67.9111099243164, 16.80000114440918, 26.311111450195312], score=1.0
image_id=0, category_id=1, bbox=[187.0, 98.13333892822266, 17.0, 25.24443817138672], score=1.0
image_id=0, category_id=1, bbox=[190.1999969482422, 193.7777862548828, 20.20001220703125, 32.0], score=1.0
image_id=0, category_id=1, bbox=[37.0, 77.15555572509766, 15.799999237060547, 27.377777099609375], score=1.0
image_id=0, category_id=1, bbox=[160.1999969482422, 172.08889770507812, 17.400009155273438, 29.155548095703125], score=1.0
image_id=0, category_id=1, bbox=[53.60000228881836, 64.0, 19.999996185302734, 35.911109924316406], score=1.0
image_id=0, category_id=1, bbox=[126.5999984741211, 146.4888916015625, 14.200004577636719, 27.377777099609375], score=1.0
image_id=0, category_id=1, bbox=[140.0, 33.77777862548828, 14.600006103515625, 24.177780151367188], score=1.0
image_id=0, category_id=1, bbox=[208.8000030517578, 204.44444274902344, 17.399993896484375, 27.377792358398438], score=1.0
image_id=0, category_id=1, bbox=[219.0, 15.644444465637207, 18.600006103515625, 29.155555725097656], score=1.0
image_id=0, category_id=1, bbox=[88.20000457763672, 166.40000915527344, 14.199996948242188, 23.466659545898438], score=1.0
image_id=0, category_id=1, bbox=[191.8000030517578, 120.8888931274414, 14.800003051757812, 22.400001525878906], score=1.0
image_id=0, category_id=1, bbox=[62.60000228881836, 216.53334045410156, 18.60000228881836, 32.71110534667969], score=1.0
image_id=0, category_id=1, bbox=[43.400001525878906, 143.6444549560547, 14.200000762939453, 21.688888549804688], score=1.0
image_id=0, category_id=1, bbox=[178.40000915527344, 65.77777862548828, 15.79998779296875, 25.95555877685547], score=1.0
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1.30s).
Accumulating evaluation results...
DONE (t=0.03s).

=== Sanity mAP summary ===
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
entering training
