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

if __name__ == "__main__":
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np
    import torch

    dataset = val_dataset  # your validation dataset

    # Prepare dummy predictions using GT boxes
    predictions = []
    for img_idx in range(len(dataset)):
        image, target = dataset[img_idx]
        # target['boxes'] is [N,4] resized to 256x256
        # We'll pretend the model predicted exactly GT boxes with score 1.0
        predictions.append({
            'boxes': target['boxes'],           # boxes in resized image coords
            'scores': torch.ones(len(target['boxes'])),  # dummy score
            'labels': target['labels']          # same class labels
        })

    coco_gt = dataset.coco
    scaled_predictions = []

    print("Running sanity check for mAP calculation using GT boxes as predictions...")

    for pred, (image, target) in zip(predictions, [dataset[i] for i in range(len(dataset))]):
        img_id = target['image_id'].item()  # integer
        img_info = coco_gt.loadImgs(img_id)[0]
        w_orig, h_orig = img_info['width'], img_info['height']

        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()

        # Dataset resize
        w_new, h_new = dataset.resize
        scale_x = w_orig / w_new
        scale_y = h_orig / h_new

        print(f"\n=== Image {img_id} ===")
        print(f"Original size: ({w_orig}, {h_orig}), Resized: ({w_new}, {h_new})")
        print(f"Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")
        print("Pred boxes before scaling (first 5):")
        print(boxes[:5])

        # Scale boxes back to original image size
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        print("Pred boxes after scaling (first 5):")
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

    print("\nLoading and preparing COCO results...")
    coco_dt = coco_gt.loadRes(scaled_predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

    print("Running per-image evaluation...")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

import copy
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import numpy as np

# --- 1. Fix category IDs in GT ---
base_dataset = val_dataset
cocoGt = copy.deepcopy(base_dataset.coco)

for ann in cocoGt.dataset['annotations']:
    if ann['category_id'] in [1, 2]:
        ann['category_id'] = 1
cocoGt.dataset['categories'] = [{"id": 1, "name": "mushroom"}]
cocoGt.createIndex()

# --- 2. Prepare predictions using GT boxes ---
predictions = []
for img_idx in range(len(base_dataset)):
    image, target = base_dataset[img_idx]
    predictions.append({
        'boxes': target['boxes'],                  # resized image coords
        'scores': torch.ones(len(target['boxes'])),  # dummy score 1.0
        'labels': torch.ones(len(target['boxes']))   # set to 1 to match cocoGt
    })

# --- 3. Convert predictions to COCO format with scaling ---
scaled_predictions = []
for pred, (image, target) in zip(predictions, [base_dataset[i] for i in range(len(base_dataset))]):
    img_id = target['image_id'].item()
    img_info = cocoGt.loadImgs(img_id)[0]
    w_orig, h_orig = img_info['width'], img_info['height']

    w_new, h_new = base_dataset.resize
    scale_x = w_orig / w_new
    scale_y = h_orig / h_new

    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()

    # debug prints before scaling
    print(f"\n=== Image {img_id} ===")
    print(f"Original size: ({w_orig}, {h_orig}), Resized: ({w_new}, {h_new})")
    print("Boxes before scaling (first 5):")
    print(boxes[:5])
    print("Labels (first 5):", labels[:5])
    
    # scale back to original image size
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    # debug prints after scaling
    print("Boxes after scaling (first 5):")
    print(boxes[:5])

    # convert to COCO bbox format [x, y, width, height]
    coco_boxes = []
    for b in boxes:
        x, y, x2, y2 = b
        width, height = x2 - x, y2 - y
        coco_boxes.append([x, y, width, height])
        # sanity check
        if width <= 0 or height <= 0:
            print(f"WARNING: Box with non-positive width/height: {b}")

    for b, s, l in zip(coco_boxes, scores, labels):
        scaled_predictions.append({
            "image_id": img_id,
            "category_id": int(l),
            "bbox": [float(coord) for coord in b],
            "score": float(s)
        })

# --- 4. Run COCO evaluation ---
coco_dt = cocoGt.loadRes(scaled_predictions)
coco_eval = COCOeval(cocoGt, coco_dt, iouType='bbox')

print("\nRunning per-image evaluation...")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# --- 5. Extra debug: show images with AP < 1 ---
for img_id in coco_eval.evalImgs:
    eval_info = coco_eval.evalImgs[img_id]
    if eval_info is not None:
        precision = eval_info['precision']  # [TxRxKxAxM]
        if precision is not None:
            mean_prec = np.mean(precision)
            if mean_prec < 1.0:
                print(f"Image {img_id} has mean precision < 1: {mean_prec:.4f}")



 [102.6      147.55556  120.4      181.33334 ]
 [151.2       91.37778  169.2      122.31111 ]
 [ 77.6       55.466667  93.        81.77778 ]
 [113.4      111.288895 135.40001  151.82222 ]]
Labels (first 5): [1. 1. 1. 1. 1.]
Boxes after scaling (first 5):
[[276.      143.      389.      244.     ]
 [513.      415.      602.      510.00003]
 [756.      257.      846.      344.     ]
 [388.      156.      465.      230.     ]
 [567.      313.00003 677.00006 427.     ]]

=== Image 436 ===
Original size: (1280, 720), Resized: (256, 256)
Boxes before scaling (first 5):
[[186.40001  187.37778  209.8      227.55556 ]
 [  7.8      111.288895  30.6      145.06667 ]
 [201.8      130.48889  224.       165.33334 ]
 [ 89.4      223.2889   105.8      255.64445 ]
 [ 63.8       83.91111   85.       120.53333 ]]
Labels (first 5): [1. 1. 1. 1. 1.]
Boxes after scaling (first 5):
[[ 932.00006  527.      1049.       640.     ]
 [  39.       313.00003  153.       408.     ]
 [1009.       367.      1120.       465.00003]
 [ 447.       628.       529.       719.     ]
 [ 319.       236.       425.       339.     ]]

=== Image 437 ===
Original size: (1280, 720), Resized: (256, 256)
Boxes before scaling (first 5):
[[120.200005 169.24445  139.2      201.24445 ]
 [205.6      203.37778  226.2      238.22223 ]
 [ 30.6       53.68889   54.4       87.46667 ]
 [100.        15.644444 118.        48.355556]
 [ 67.4      206.22223   85.200005 239.2889  ]]
Labels (first 5): [1. 1. 1. 1. 1.]
Boxes after scaling (first 5):
[[ 601.       476.       696.       566.     ]
 [1028.       572.      1131.       670.     ]
 [ 153.       151.       272.       246.     ]
 [ 500.        44.       590.       136.     ]
 [ 337.       580.       426.00003  673.     ]]

=== Image 438 ===
Original size: (1280, 720), Resized: (256, 256)
Boxes before scaling (first 5):
[[ 24.2      183.46667   43.8      216.53334 ]
 [112.8      150.40001  130.6      183.82222 ]
 [ 75.4       23.111113  91.8       52.266666]
 [106.4       56.533333 121.8       87.46667 ]
 [198.6      136.88889  219.       170.66667 ]]
Labels (first 5): [1. 1. 1. 1. 1.]
Boxes after scaling (first 5):
[[ 121.       516.       219.       609.     ]
 [ 564.       423.00003  653.       517.     ]
 [ 377.        65.00001  459.       147.     ]
 [ 532.       159.       609.       246.     ]
 [ 993.       385.      1095.       480.     ]]

=== Image 439 ===
Original size: (1280, 720), Resized: (256, 256)
Boxes before scaling (first 5):
[[137.40001  210.84445  154.8      244.62222 ]
 [184.8       73.6      204.40001  106.66667 ]
 [118.6      192.       134.       220.0889  ]
 [232.8      193.77779  255.8      228.62222 ]
 [197.6        0.       216.        23.111113]]
Labels (first 5): [1. 1. 1. 1. 1.]
Boxes after scaling (first 5):
[[ 687.00006  593.       774.       688.     ]
 [ 924.       207.      1022.00006  300.     ]
 [ 593.       540.       670.       619.     ]
 [1164.       545.      1279.       643.     ]
 [ 988.         0.      1080.        65.00001]]

=== Image 440 ===
Original size: (1280, 720), Resized: (256, 256)
Boxes before scaling (first 5):
[[ 27.      103.82223  49.8     139.73334]
 [ 28.6      71.46667  50.4     104.53333]
 [181.6     183.46667 197.8     210.84445]
 [156.6      65.77778 174.40001  99.55556]
 [ 59.4     169.24445  77.4     199.46667]]
Labels (first 5): [1. 1. 1. 1. 1.]
Boxes after scaling (first 5):
[[135.      292.      249.      393.     ]
 [143.      201.      252.      294.     ]
 [908.      516.      989.      593.     ]
 [783.      185.      872.00006 280.     ]
 [297.      476.      387.      561.     ]]

=== Image 441 ===
Original size: (1280, 720), Resized: (256, 256)
Boxes before scaling (first 5):
[[230.6      207.2889   252.40001  241.06667 ]
 [117.6      145.42223  139.40001  187.02223 ]
 [162.40001  160.71112  182.8      197.33334 ]
 [ 41.4      151.11111   62.600002 184.8889  ]
 [ 33.4      118.04445   56.2      154.66667 ]]
Labels (first 5): [1. 1. 1. 1. 1.]
Boxes after scaling (first 5):
[[1153.       583.      1262.       678.     ]
 [ 588.       409.       697.00006  526.     ]
 [ 812.00006  452.00003  914.       555.     ]
 [ 207.       425.       313.       520.00006]
 [ 167.       332.       281.       435.     ]]

=== Image 442 ===
Original size: (1280, 720), Resized: (256, 256)
Boxes before scaling (first 5):
[[131.       155.02223  148.6      189.86667 ]
 [  0.       125.511116  17.2      154.66667 ]
 [231.2       13.866667 248.6       40.17778 ]
 [156.6      133.33334  175.2      164.26668 ]
 [ 10.400001  68.62222   37.8      108.08889 ]]
Labels (first 5): [1. 1. 1. 1. 1.]
Boxes after scaling (first 5):
[[ 655.        436.00003   743.        534.      ]
 [   0.        353.         86.        435.      ]
 [1156.         39.       1243.        113.00001 ]
 [ 783.        375.00003   876.        462.00003 ]
 [  52.000004  193.        189.        304.      ]]

=== Image 443 ===
Original size: (1280, 720), Resized: (256, 256)
Boxes before scaling (first 5):
[[ 63.2     126.57778  81.6     161.42223]
 [185.8     108.44445 206.40001 137.6    ]
 [123.        0.      140.40001  34.48889]
 [ 49.8     174.93333  65.      203.02223]
 [ 14.2     225.06667  34.8     255.64445]]
Labels (first 5): [1. 1. 1. 1. 1.]
Boxes after scaling (first 5):
[[ 316.       356.       408.       454.     ]
 [ 929.       305.00003 1032.       387.00003]
 [ 615.         0.       702.00006   97.00001]
 [ 249.       492.       325.       571.     ]
 [  71.       633.       174.       719.     ]]

=== Image 444 ===
Original size: (1280, 720), Resized: (256, 256)
Boxes before scaling (first 5):
[[183.2      123.73334  208.2      164.26668 ]
 [ 43.4      131.2       64.6      162.13333 ]
 [  0.       160.71112   18.800001 192.71112 ]
 [ 14.2      183.46667   37.600002 221.15556 ]
 [ 86.6      161.77779  102.4      189.86667 ]]
Labels (first 5): [1. 1. 1. 1. 1.]
Boxes after scaling (first 5):
[[ 916.       348.      1041.       462.00003]
 [ 217.       369.       323.       456.     ]
 [   0.       452.00003   94.00001  542.     ]
 [  71.       516.       188.00002  622.     ]
 [ 433.       455.00003  512.       534.     ]]

=== Image 445 ===
Original size: (1280, 720), Resized: (256, 256)
Boxes before scaling (first 5):
[[1.88600006e+02 1.25511116e+02 2.06000000e+02 1.56444443e+02]
 [2.34400009e+02 1.76000000e+02 2.53000000e+02 2.04088898e+02]
 [2.00000003e-01 1.67466675e+02 1.60000000e+01 2.06222229e+02]
 [2.25400009e+02 3.55555573e+01 2.48000000e+02 6.86222229e+01]
 [2.96000004e+01 2.27911118e+02 5.17999992e+01 2.55644455e+02]]
Labels (first 5): [1. 1. 1. 1. 1.]
Boxes after scaling (first 5):
[[9.4300000e+02 3.5300000e+02 1.0300000e+03 4.4000000e+02]
 [1.1720000e+03 4.9500000e+02 1.2650000e+03 5.7400000e+02]
 [1.0000000e+00 4.7100003e+02 8.0000000e+01 5.8000000e+02]
 [1.1270000e+03 1.0000001e+02 1.2400000e+03 1.9300000e+02]
 [1.4800000e+02 6.4100000e+02 2.5900000e+02 7.1900000e+02]]

=== Image 446 ===
Original size: (1280, 720), Resized: (256, 256)
Boxes before scaling (first 5):
[[152.8       81.066666 173.40001  117.68889 ]
 [131.40001  135.11111  146.8      164.26668 ]
 [175.2       93.511116 190.6      121.600006]
 [  9.400001 180.62222   25.800001 204.0889  ]
 [ 69.6       56.533333  85.        81.77778 ]]
Labels (first 5): [1. 1. 1. 1. 1.]
Boxes after scaling (first 5):
[[764.       228.       867.00006  331.      ]
 [657.00006  380.       734.       462.00003 ]
 [876.       263.       953.       342.00003 ]
 [ 47.000004 508.       129.       574.      ]
 [348.       159.       425.       230.      ]]

=== Image 447 ===
Original size: (1280, 720), Resized: (256, 256)
Boxes before scaling (first 5):
[[208.8       93.511116 231.40001  128.35556 ]
 [ 80.200005  68.62222   98.8       99.200005]
 [ 97.4      137.95557  114.4      169.95557 ]
 [222.6      142.57777  242.2      171.73334 ]
 [140.       145.42223  160.6      178.48889 ]]
Labels (first 5): [1. 1. 1. 1. 1.]
Boxes after scaling (first 5):
[[1044.       263.      1157.       361.     ]
 [ 401.00003  193.       494.       279.     ]
 [ 487.       388.00003  572.       478.00003]
 [1113.       401.      1211.       483.     ]
 [ 700.       409.       803.       502.     ]]

=== Image 448 ===
Original size: (1280, 720), Resized: (256, 256)
Boxes before scaling (first 5):
[[173.6       99.200005 194.8      134.75555 ]
 [134.2      176.       152.8      210.84445 ]
 [231.2      169.24445  248.6      196.62222 ]
 [105.4       62.933334 130.2      102.4     ]
 [ 48.8      178.84445   69.4      216.53334 ]]
Labels (first 5): [1. 1. 1. 1. 1.]
Boxes after scaling (first 5):
[[ 868.       279.       974.       379.     ]
 [ 671.       495.       764.       593.     ]
 [1156.       476.      1243.       553.     ]
 [ 527.       177.       651.       288.     ]
 [ 244.       503.00003  347.       609.     ]]
Loading and preparing results...
DONE (t=2.00s)
creating index...
index created!

Running per-image evaluation...
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=118.36s).
Accumulating evaluation results...
DONE (t=0.37s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.614
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.614
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.614
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.109
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.644
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.960
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.006
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.063
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.619
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.100
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.644
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.962
Traceback (most recent call last):
  File "/home/matson/mushroom-mask-rcnn/train_model.py", line 303, in <module>
    eval_info = coco_eval.evalImgs[img_id]
TypeError: list indices must be integers or slices, not dict


