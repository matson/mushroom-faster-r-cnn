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

    # Use your validation dataset
    dataset = val_dataset  # exactly the variable you defined

    # Create dummy predictions for sanity check (same size as val_dataset)
    # Each pred is a dict: 'boxes', 'scores', 'labels'
    # We'll just make small boxes to simulate predictions
    predictions = []
    for image, target in dataset:
        # get the image_id
        img_id = int(target['image_id'])
        # Make dummy boxes: same as target boxes but scaled down
        boxes = target['boxes'] * 0.2  # simulate scaling issue
        scores = torch.ones(len(boxes))
        labels = target['labels']
        predictions.append({
            "boxes": boxes,
            "scores": scores,
            "labels": labels
        })

    # --- Sanity check / mAP scaling ---
    coco_gt = dataset.coco
    scaled_predictions = []

    print("Running sanity check for mAP calculation...\n")

    for pred, target in zip(predictions, dataset):
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        img_id = int(target['image_id'])
        img_info = coco_gt.loadImgs(img_id)[0]
        w_orig, h_orig = img_info['width'], img_info['height']
        w_new, h_new = dataset.resize
        scale_x = w_orig / w_new
        scale_y = h_orig / h_new

        print(f"\n=== Image {img_id} ===")
        print(f"Original size: ({w_orig}, {h_orig}), Resized: ({w_new}, {h_new})")
        print(f"Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")
        print("Pred boxes before scaling (first 5):")
        print(boxes[:5])

        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        print("Pred boxes after scaling (first 5):")
        print(boxes[:5])

        # Show GT boxes for comparison
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        gt_boxes = []
        gt_labels = []
        for ann in anns:
            x, y, width, height = ann['bbox']
            gt_boxes.append([x, y, x + width, y + height])
            gt_labels.append(ann['category_id'])
        gt_boxes = np.array(gt_boxes)
        gt_labels = np.array(gt_labels)
        print("GT boxes (first 5):")
        print(gt_boxes[:5])
        print("GT labels (first 5):")
        print(gt_labels[:5])

        # Convert to COCO format
        coco_boxes = [[float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])] for b in boxes]

        for b, s, l in zip(coco_boxes, scores, labels):
            scaled_predictions.append({
                "image_id": img_id,
                "category_id": int(l),
                "bbox": b,
                "score": float(s)
            })

    print("\nLoading and preparing COCO results...")
    coco_dt = coco_gt.loadRes(scaled_predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

    print("Running per-image evaluation...")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np
    import torch

    # Use your validation dataset
    dataset = val_dataset  # exactly the variable you defined

    # Create dummy predictions for sanity check (same size as val_dataset)
    # Each pred is a dict: 'boxes', 'scores', 'labels'
    # We'll just make small boxes to simulate predictions
    predictions = []
    for image, target in dataset:
        # get the image_id
        img_id = int(target['image_id'])
        # Make dummy boxes: same as target boxes but scaled down
        boxes = target['boxes'] * 0.2  # simulate scaling issue
        scores = torch.ones(len(boxes))
        labels = target['labels']
        predictions.append({
            "boxes": boxes,
            "scores": scores,
            "labels": labels
        })

    # --- Sanity check / mAP scaling ---
    coco_gt = dataset.coco
    scaled_predictions = []

    print("Running sanity check for mAP calculation...\n")

    for pred, target in zip(predictions, dataset):
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        img_id = int(target['image_id'])
        img_info = coco_gt.loadImgs(img_id)[0]
        w_orig, h_orig = img_info['width'], img_info['height']
        w_new, h_new = dataset.resize
        scale_x = w_orig / w_new
        scale_y = h_orig / h_new

        print(f"\n=== Image {img_id} ===")
        print(f"Original size: ({w_orig}, {h_orig}), Resized: ({w_new}, {h_new})")
        print(f"Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")
        print("Pred boxes before scaling (first 5):")
        print(boxes[:5])

        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        print("Pred boxes after scaling (first 5):")
        print(boxes[:5])

        # Show GT boxes for comparison
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        gt_boxes = []
        gt_labels = []
        for ann in anns:
            x, y, width, height = ann['bbox']
            gt_boxes.append([x, y, x + width, y + height])
            gt_labels.append(ann['category_id'])
        gt_boxes = np.array(gt_boxes)
        gt_labels = np.array(gt_labels)
        print("GT boxes (first 5):")
        print(gt_boxes[:5])
        print("GT labels (first 5):")
        print(gt_labels[:5])

        # Convert to COCO format
        coco_boxes = [[float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])] for b in boxes]

        for b, s, l in zip(coco_boxes, scores, labels):
            scaled_predictions.append({
                "image_id": img_id,
                "category_id": int(l),
                "bbox": b,
                "score": float(s)
            })

    print("\nLoading and preparing COCO results...")
    coco_dt = coco_gt.loadRes(scaled_predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

    print("Running per-image evaluation...")
    coco_eval.evaluate()

Pred boxes after scaling (first 5):
[[ 55.2       28.6       77.8       48.8     ]
 [102.600006  83.       120.4      102.00001 ]
 [151.2       51.4      169.2       68.8     ]
 [ 77.6       31.199999  93.        46.000004]
 [113.4       62.600006 135.40001   85.4     ]]
GT boxes (first 5):
[[276 143 389 244]
 [513 415 602 510]
 [756 257 846 344]
 [388 156 465 230]
 [567 313 677 427]]
GT labels (first 5):
[2 2 2 2 2]

=== Image 436 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[37.280003  37.475555  41.960003  45.511112 ]
 [ 1.5600001 22.25778    6.1200004 29.013334 ]
 [40.36      26.097778  44.8       33.06667  ]
 [17.880001  44.65778   21.160002  51.12889  ]
 [12.76      16.782223  17.        24.106667 ]]
Pred boxes after scaling (first 5):
[[186.40001  105.4      209.80002  128.      ]
 [  7.8       62.600006  30.600002  81.600006]
 [201.8       73.4      224.        93.00001 ]
 [ 89.40001  125.600006 105.80001  143.8     ]
 [ 63.800003  47.2       85.        67.8     ]]
GT boxes (first 5):
[[ 932  527 1049  640]
 [  39  313  153  408]
 [1009  367 1120  465]
 [ 447  628  529  719]
 [ 319  236  425  339]]
GT labels (first 5):
[2 2 2 2 2]

=== Image 437 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[24.04      33.84889   27.84      40.24889  ]
 [41.120003  40.675556  45.24      47.644447 ]
 [ 6.1200004 10.737778  10.88      17.493334 ]
 [20.         3.1288888 23.6        9.671111 ]
 [13.4800005 41.244446  17.04      47.85778  ]]
Pred boxes after scaling (first 5):
[[120.200005  95.2      139.2      113.200005]
 [205.6      114.4      226.20001  134.00002 ]
 [ 30.600002  30.2       54.4       49.2     ]
 [100.         8.8      118.        27.2     ]
 [ 67.4      116.        85.200005 134.6     ]]
GT boxes (first 5):
[[ 601  476  696  566]
 [1028  572 1131  670]
 [ 153  151  272  246]
 [ 500   44  590  136]
 [ 337  580  426  673]]
GT labels (first 5):
[2 2 2 2 2]

=== Image 438 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[ 4.84      36.693336   8.76      43.306667 ]
 [22.560001  30.080002  26.12      36.764446 ]
 [15.080001   4.6222224 18.36      10.453334 ]
 [21.28      11.306666  24.36      17.493334 ]
 [39.72      27.377777  43.8       34.133335 ]]
Pred boxes after scaling (first 5):
[[ 24.2      103.20001   43.800003 121.8     ]
 [112.8       84.600006 130.6      103.4     ]
 [ 75.4       13.000001  91.8       29.400002]
 [106.4       31.8      121.8       49.2     ]
 [198.6       77.       219.        96.00001 ]]
GT boxes (first 5):
[[ 121  516  219  609]
 [ 564  423  653  517]
 [ 377   65  459  147]
 [ 532  159  609  246]
 [ 993  385 1095  480]]
GT labels (first 5):
[2 2 2 2 2]

=== Image 439 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[27.480001  42.16889   30.960001  48.924446 ]
 [36.960003  14.72      40.88      21.333334 ]
 [23.72      38.4       26.800001  44.01778  ]
 [46.56      38.755558  51.16      45.724445 ]
 [39.52       0.        43.2        4.6222224]]
Pred boxes after scaling (first 5):
[[137.40001  118.600006 154.8      137.6     ]
 [184.80002   41.4      204.40001   60.      ]
 [118.6      108.00001  134.       123.80001 ]
 [232.8      109.00001  255.8      128.6     ]
 [197.6        0.       216.        13.000001]]
GT boxes (first 5):
[[ 687  593  774  688]
 [ 924  207 1022  300]
 [ 593  540  670  619]
 [1164  545 1279  643]
 [ 988    0 1080   65]]
GT labels (first 5):
[2 2 2 2 2]

=== Image 440 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[ 5.4       20.764446   9.96      27.946669 ]
 [ 5.7200003 14.293334  10.080001  20.906668 ]
 [36.320004  36.693336  39.56      42.16889  ]
 [31.320002  13.155556  34.88      19.911112 ]
 [11.88      33.84889   15.4800005 39.893337 ]]
Pred boxes after scaling (first 5):
[[ 27.        58.400005  49.8       78.600006]
 [ 28.600002  40.2       50.400005  58.800003]
 [181.60002  103.20001  197.8      118.600006]
 [156.6       37.       174.40001   56.000004]
 [ 59.4       95.2       77.4      112.20001 ]]
GT boxes (first 5):
[[135 292 249 393]
 [143 201 252 294]
 [908 516 989 593]
 [783 185 872 280]
 [297 476 387 561]]
GT labels (first 5):
[2 2 2 2 2]

=== Image 441 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[46.120003  41.45778   50.480003  48.213333 ]
 [23.52      29.084446  27.880003  37.404446 ]
 [32.480003  32.142223  36.56      39.46667  ]
 [ 8.280001  30.222223  12.52      36.97778  ]
 [ 6.6800003 23.60889   11.240001  30.933334 ]]
Pred boxes after scaling (first 5):
[[230.6      116.600006 252.40002  135.6     ]
 [117.600006  81.8      139.40001  105.200005]
 [162.40002   90.4      182.8      111.000015]
 [ 41.4       85.        62.600002 104.00001 ]
 [ 33.4       66.4       56.200005  87.      ]]
GT boxes (first 5):
[[1153  583 1262  678]
 [ 588  409  697  526]
 [ 812  452  914  555]
 [ 207  425  313  520]
 [ 167  332  281  435]]
GT labels (first 5):
[2 2 2 2 2]

=== Image 442 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[26.2       31.004446  29.720001  37.973335 ]
 [ 0.        25.102224   3.4400003 30.933334 ]
 [46.24       2.7733333 49.72       8.035556 ]
 [31.320002  26.66667   35.04      32.853336 ]
 [ 2.0800002 13.724444   7.56      21.617779 ]]
Pred boxes after scaling (first 5):
[[131.         87.200005  148.6       106.8      ]
 [  0.         70.600006   17.2        87.       ]
 [231.20001     7.7999997 248.6        22.6      ]
 [156.6        75.00001   175.20001    92.40001  ]
 [ 10.400001   38.6        37.8        60.800003 ]]
GT boxes (first 5):
[[ 655  436  743  534]
 [   0  353   86  435]
 [1156   39 1243  113]
 [ 783  375  876  462]
 [  52  193  189  304]]
GT labels (first 5):
[2 2 2 2 2]

=== Image 443 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[12.64      25.315557  16.32      32.284447 ]
 [37.16      21.68889   41.280003  27.520002 ]
 [24.6        0.        28.080002   6.8977785]
 [ 9.96      34.986668  13.        40.604446 ]
 [ 2.84      45.013332   6.96      51.12889  ]]
Pred boxes after scaling (first 5):
[[ 63.2       71.200005  81.6       90.8     ]
 [185.8       61.000004 206.40001   77.40001 ]
 [123.         0.       140.40001   19.400002]
 [ 49.8       98.4       65.       114.200005]
 [ 14.2      126.6       34.8      143.8     ]]
GT boxes (first 5):
[[ 316  356  408  454]
 [ 929  305 1032  387]
 [ 615    0  702   97]
 [ 249  492  325  571]
 [  71  633  174  719]]
GT labels (first 5):
[2 2 2 2 2]

=== Image 444 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[36.64      24.746668  41.64      32.853336 ]
 [ 8.68      26.24      12.92      32.426666 ]
 [ 0.        32.142223   3.7600002 38.542225 ]
 [ 2.84      36.693336   7.5200005 44.231113 ]
 [17.32      32.355556  20.480001  37.973335 ]]
Pred boxes after scaling (first 5):
[[183.2       69.600006 208.2       92.40001 ]
 [ 43.4       73.8       64.6       91.2     ]
 [  0.        90.4       18.800001 108.40001 ]
 [ 14.2      103.20001   37.600002 124.40001 ]
 [ 86.6       91.       102.40001  106.8     ]]
GT boxes (first 5):
[[ 916  348 1041  462]
 [ 217  369  323  456]
 [   0  452   94  542]
 [  71  516  188  622]
 [ 433  455  512  534]]
GT labels (first 5):
[2 2 2 2 2]

=== Image 445 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[3.7720001e+01 2.5102224e+01 4.1200001e+01 3.1288889e+01]
 [4.6880001e+01 3.5200001e+01 5.0600002e+01 4.0817780e+01]
 [4.0000003e-02 3.3493336e+01 3.2000000e+00 4.1244446e+01]
 [4.5080002e+01 7.1111116e+00 4.9600002e+01 1.3724444e+01]
 [5.9200001e+00 4.5582226e+01 1.0360000e+01 5.1128891e+01]]
Pred boxes after scaling (first 5):
[[1.8860001e+02 7.0600006e+01 2.0600000e+02 8.8000000e+01]
 [2.3440001e+02 9.9000000e+01 2.5300002e+02 1.1480000e+02]
 [2.0000002e-01 9.4200005e+01 1.6000000e+01 1.1600000e+02]
 [2.2540001e+02 2.0000002e+01 2.4800002e+02 3.8599998e+01]
 [2.9600000e+01 1.2820001e+02 5.1799999e+01 1.4380000e+02]]
GT boxes (first 5):
[[ 943  353 1030  440]
 [1172  495 1265  574]
 [   1  471   80  580]
 [1127  100 1240  193]
 [ 148  641  259  719]]
GT labels (first 5):
[2 2 2 2 2]

=== Image 446 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[30.560001  16.213333  34.680004  23.537779 ]
 [26.280003  27.022223  29.36      32.853336 ]
 [35.04      18.702223  38.120003  24.320002 ]
 [ 1.8800001 36.124447   5.1600003 40.81778  ]
 [13.92      11.306666  17.        16.355556 ]]
Pred boxes after scaling (first 5):
[[152.8       45.6      173.40002   66.200005]
 [131.40001   76.       146.8       92.40001 ]
 [175.20001   52.600002 190.6       68.4     ]
 [  9.400001 101.600006  25.800001 114.8     ]
 [ 69.6       31.8       85.        46.000004]]
GT boxes (first 5):
[[764 228 867 331]
 [657 380 734 462]
 [876 263 953 342]
 [ 47 508 129 574]
 [348 159 425 230]]
GT labels (first 5):
[2 2 2 2 2]

=== Image 447 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[41.760002 18.702223 46.280003 25.671112]
 [16.04     13.724444 19.76     19.840002]
 [19.480001 27.591114 22.880001 33.991116]
 [44.52     28.515554 48.44     34.34667 ]
 [28.       29.084446 32.120003 35.69778 ]]
Pred boxes after scaling (first 5):
[[208.80002   52.600002 231.40001   72.200005]
 [ 80.200005  38.6       98.8       55.800007]
 [ 97.40001   77.600006 114.40001   95.60001 ]
 [222.6       80.2      242.2       96.600006]
 [140.        81.8      160.6      100.40001 ]]
GT boxes (first 5):
[[1044  263 1157  361]
 [ 401  193  494  279]
 [ 487  388  572  478]
 [1113  401 1211  483]
 [ 700  409  803  502]]
GT labels (first 5):
[2 2 2 2 2]

=== Image 448 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[34.72     19.840002 38.960003 26.95111 ]
 [26.84     35.2      30.560001 42.16889 ]
 [46.24     33.84889  49.72     39.324444]
 [21.08     12.586667 26.039999 20.480001]
 [ 9.76     35.76889  13.88     43.306667]]
Pred boxes after scaling (first 5):
[[173.6       55.800007 194.80002   75.8     ]
 [134.2       99.       152.8      118.600006]
 [231.20001   95.2      248.6      110.6     ]
 [105.4       35.4      130.2       57.600002]
 [ 48.800003 100.600006  69.4      121.8     ]]
GT boxes (first 5):
[[ 868  279  974  379]
 [ 671  495  764  593]
 [1156  476 1243  553]
 [ 527  177  651  288]
 [ 244  503  347  609]]
GT labels (first 5):
[2 2 2 2 2]

Loading and preparing COCO results...
Loading and preparing results...
DONE (t=0.09s)
creating index...
index created!
Running per-image evaluation...
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1.03s).
Accumulating evaluation results...
DONE (t=0.31s).
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
    coco_eval.accumulate()
    coco_eval.summarize()

