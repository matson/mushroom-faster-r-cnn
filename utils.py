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

 [ 988.       631.      1086.       720.     ]]

=== Image 413 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[101.        82.13334  117.4      108.8     ]
 [ 51.4      112.35556   75.6      151.46667 ]
 [ 93.6      135.11111  112.6      166.04445 ]
 [ 97.8      226.13333  118.4      256.      ]
 [ 78.200005 178.84445   96.8      216.53334 ]]
Pred boxes after scaling (first 5):
[[505.      231.00002 587.      306.     ]
 [257.      316.      378.      426.00003]
 [468.      380.      563.      467.     ]
 [489.      636.      592.      720.     ]
 [391.00003 503.00003 484.      609.     ]]

=== Image 414 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[ 50.4      178.84445   74.200005 216.53334 ]
 [203.40001  192.       225.       221.15556 ]
 [231.8      203.37778  253.       236.44444 ]
 [189.6      118.04445  210.2      148.97778 ]
 [ 34.4       80.        58.8      120.53333 ]]
Pred boxes after scaling (first 5):
[[ 252.       503.00003  371.00003  609.     ]
 [1017.00006  540.      1125.       622.     ]
 [1159.       572.      1265.       665.     ]
 [ 948.       332.      1051.       419.     ]
 [ 172.       225.       294.       339.     ]]

=== Image 415 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[ 46.2       57.244446  63.2       88.17778 ]
 [172.       194.84445  191.6      229.33334 ]
 [139.        69.68889  157.40001  103.46667 ]
 [171.        45.86667  189.6       78.933334]
 [198.6       20.266666 223.40001   54.044445]]
Pred boxes after scaling (first 5):
[[ 231.       161.       316.       248.     ]
 [ 860.       548.       958.       645.     ]
 [ 695.       196.       787.00006  291.     ]
 [ 855.       129.       948.       222.     ]
 [ 993.        57.      1117.       152.     ]]

=== Image 416 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[210.40001  139.02223  234.8      172.44444 ]
 [166.2      198.75555  184.2      231.46667 ]
 [181.6       97.066666 199.6      126.22223 ]
 [222.6      116.26667  255.8      167.11111 ]
 [215.8      220.44444  233.2      248.53334 ]]
Pred boxes after scaling (first 5):
[[1052.       391.00003 1174.       485.     ]
 [ 831.       559.       921.       651.     ]
 [ 908.       273.       998.       355.00003]
 [1113.       327.      1279.       470.     ]
 [1079.       620.      1166.       699.     ]]

=== Image 417 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[186.40001   135.11111   203.40001   163.2      ]
 [ 67.4       116.97778    82.8       144.35556  ]
 [ 93.6        48.        114.200005   85.68889  ]
 [  6.2000003 139.02223    23.6       167.11111  ]
 [148.6       214.75555   164.        243.91112  ]]
Pred boxes after scaling (first 5):
[[ 932.00006   380.       1017.00006   459.      ]
 [ 337.        329.        414.        406.      ]
 [ 468.        135.        571.        241.      ]
 [  31.000002  391.00003   118.        470.      ]
 [ 743.        604.        820.        686.      ]]

=== Image 418 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[127.200005 122.66667  146.8      162.13333 ]
 [ 27.4      182.40001   53.4      222.93333 ]
 [169.8      119.111115 197.40001  163.2     ]
 [ 85.       145.42223  101.4      177.42223 ]
 [231.2      161.77779  255.8      195.55556 ]]
Pred boxes after scaling (first 5):
[[ 636.       345.       734.       456.     ]
 [ 137.       513.       267.       627.     ]
 [ 849.       335.       987.00006  459.     ]
 [ 425.       409.       507.       499.     ]
 [1156.       455.00003 1279.       550.     ]]

=== Image 419 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[212.6      140.8      238.6      178.48889 ]
 [127.200005  92.44445  142.6      120.53333 ]
 [  8.8       96.35556   33.2      127.288895]
 [ 73.8      108.44445   90.8      137.6     ]
 [ 96.8       59.37778  114.200005  89.6     ]]
Pred boxes after scaling (first 5):
[[1063.       396.      1193.       502.     ]
 [ 636.       260.00003  713.       339.     ]
 [  44.       271.       166.       358.00003]
 [ 369.       305.00003  454.       387.00003]
 [ 484.       167.00002  571.       252.     ]]

=== Image 420 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[107.        155.02223   125.6       189.86667  ]
 [234.40001   156.8       253.40001   184.8889   ]
 [ 92.6       159.64445   110.6       199.11111  ]
 [138.40001   169.24445   155.8       200.17778  ]
 [  7.2000003  13.866667   25.2        40.17778  ]]
Pred boxes after scaling (first 5):
[[ 535.       436.00003  628.       534.     ]
 [1172.       441.      1267.       520.00006]
 [ 463.       449.00003  553.       560.     ]
 [ 692.00006  476.       779.       563.     ]
 [  36.        39.       126.       113.00001]]

=== Image 421 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[ 62.2      105.600006  79.6      133.68889 ]
 [232.8       27.022223 255.8       61.86667 ]
 [172.6      203.37778  190.6      234.31111 ]
 [ 14.2       48.711113  31.2       76.08889 ]
 [203.40001  120.88889  229.40001  159.2889  ]]
Pred boxes after scaling (first 5):
[[ 311.       297.00003  398.       376.     ]
 [1164.        76.      1279.       174.     ]
 [ 863.       572.       953.       659.     ]
 [  71.       137.       156.       214.     ]
 [1017.00006  340.      1147.       448.00003]]

=== Image 422 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[110.200005 226.13333  127.6      255.64445 ]
 [ 86.200005 106.66667  110.       146.84445 ]
 [ 23.800001  45.86667   45.8       78.57778 ]
 [176.8      140.8      201.8      185.95557 ]
 [216.2      153.95557  240.8      194.84445 ]]
Pred boxes after scaling (first 5):
[[ 551.       636.       638.       719.     ]
 [ 431.00003  300.       550.       413.00003]
 [ 119.00001  129.       229.       221.00002]
 [ 884.       396.      1009.       523.     ]
 [1081.       433.00003 1204.       548.     ]]

=== Image 423 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[ 97.4      111.288895 122.6      160.35556 ]
 [ 63.8       58.31111   87.200005  93.155556]
 [129.40001   56.533333 148.40001   88.53333 ]
 [ 96.8       33.77778  116.4       74.31111 ]
 [ 98.4      162.48889  117.4      195.55556 ]]
Pred boxes after scaling (first 5):
[[487.      313.00003 613.      451.     ]
 [319.      164.      436.00003 262.     ]
 [647.00006 159.      742.00006 249.     ]
 [484.       95.      582.      209.     ]
 [492.      457.      587.      550.     ]]

=== Image 424 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[152.8      172.0889   169.6      203.02223 ]
 [103.200005 199.46667  119.       228.62222 ]
 [192.8      151.11111  219.40001  190.57779 ]
 [169.40001  144.71112  188.       176.71112 ]
 [151.2       11.733334 169.6       48.355556]]
Pred boxes after scaling (first 5):
[[ 764.       484.00003  848.       571.     ]
 [ 516.       561.       595.       643.     ]
 [ 964.       425.      1097.       536.00006]
 [ 847.00006  407.00003  940.       497.00003]
 [ 756.        33.       848.       136.     ]]

=== Image 425 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[217.8       19.555555 237.40001   47.644447]
 [ 39.8      190.22223   60.2      223.2889  ]
 [173.6        3.2      194.8       35.2     ]
 [ 41.8       40.17778   66.6       77.86667 ]
 [ 15.2      195.91112   39.4      229.68889 ]]
Pred boxes after scaling (first 5):
[[1089.        55.      1187.       134.00002]
 [ 199.       535.       301.       628.     ]
 [ 868.         9.       974.        99.     ]
 [ 209.       113.00001  333.       219.     ]
 [  76.       551.       197.       646.     ]]

=== Image 426 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[171.       193.06667  190.       221.15556 ]
 [ 61.        25.955557  79.6       57.95556 ]
 [161.8      160.71112  181.40001  197.33334 ]
 [ 80.200005  30.933334  98.        66.844444]
 [ 93.        55.466667 110.        83.55556 ]]
Pred boxes after scaling (first 5):
[[855.      543.      950.      622.     ]
 [305.       73.      398.      163.00002]
 [809.      452.00003 907.00006 555.     ]
 [401.00003  87.      490.      188.     ]
 [465.      156.      550.      235.     ]]

=== Image 427 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[ 53.        190.93333    87.        241.77779  ]
 [ 77.        174.93333   100.8       214.40001  ]
 [  7.2000003 225.06667    29.800001  255.64445  ]
 [ 86.6        80.        107.200005  108.08889  ]
 [ 26.4       141.86667    44.2       169.95557  ]]
Pred boxes after scaling (first 5):
[[265.      537.      435.      680.     ]
 [385.      492.      504.      603.     ]
 [ 36.      633.      149.      719.     ]
 [433.      225.      536.      304.     ]
 [132.      399.      221.      478.00003]]

=== Image 428 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[220.       220.44444  245.       256.      ]
 [118.6      135.11111  137.6      171.73334 ]
 [232.2       40.17778  253.40001   69.333336]
 [169.8      150.40001  186.8      181.33334 ]
 [192.8      167.46667  212.2      198.40001 ]]
Pred boxes after scaling (first 5):
[[1100.       620.      1225.       720.     ]
 [ 593.       380.       688.       483.     ]
 [1161.       113.00001 1267.       195.     ]
 [ 849.       423.00003  934.       510.00003]
 [ 964.       471.00003 1061.       558.     ]]

=== Image 429 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[2.0000000e-01 1.0488889e+02 1.8800001e+01 1.3937778e+02]
 [1.1540000e+02 1.9377779e+02 1.3680000e+02 2.3253334e+02]
 [5.1000000e+01 1.1697778e+02 7.6000000e+01 1.5644444e+02]
 [1.7460001e+02 1.4755556e+02 1.9580000e+02 1.8524445e+02]
 [1.6800001e+01 2.4177778e+01 3.6799999e+01 5.7955559e+01]]
Pred boxes after scaling (first 5):
[[  1.      295.       94.00001 392.     ]
 [577.      545.      684.      654.     ]
 [255.      329.      380.      440.     ]
 [873.      415.      979.      521.     ]
 [ 84.00001  68.      184.      163.00002]]

=== Image 430 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[207.2     225.06667 230.6     256.     ]
 [ 91.4      73.6     111.4     108.44445]
 [207.2     184.53334 226.2     214.75555]
 [180.      151.11111 195.40001 175.64445]
 [153.8     193.06667 168.6     223.2889 ]]
Pred boxes after scaling (first 5):
[[1036.       633.      1153.       720.     ]
 [ 457.       207.       557.       305.00003]
 [1036.       519.      1131.       604.     ]
 [ 900.       425.       977.00006  494.00003]
 [ 769.       543.       843.       628.     ]]

=== Image 431 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[151.8      119.111115 170.8      151.11111 ]
 [223.8      143.64445  242.8      172.44444 ]
 [182.6       86.755554 201.6      118.75556 ]
 [173.        65.77778  190.        97.77778 ]
 [183.2      153.24445  206.6      193.77779 ]]
Pred boxes after scaling (first 5):
[[ 759.       335.       854.       425.     ]
 [1119.       404.00003 1214.       485.     ]
 [ 913.       244.      1008.       334.00003]
 [ 865.       185.       950.       275.     ]
 [ 916.       431.      1033.       545.     ]]

=== Image 432 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[125.        62.222225 147.6      100.97778 ]
 [ 70.200005  20.266666  92.4       55.11111 ]
 [123.       218.66667  141.6      249.6     ]
 [155.       136.88889  174.       167.82222 ]
 [181.       165.33334  205.8      200.8889  ]]
Pred boxes after scaling (first 5):
[[ 625.       175.00002  738.       284.     ]
 [ 351.00003   57.       462.       155.     ]
 [ 615.       615.       708.       702.     ]
 [ 775.       385.       870.       472.     ]
 [ 905.       465.00003 1029.       565.00006]]

=== Image 433 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[159.2     146.48889 177.2     178.48889]
 [218.40001 126.57778 236.8     154.66667]
 [133.6      78.22222 149.      104.53333]
 [ 47.2     187.37778  64.6     212.62222]
 [187.40001 213.68889 203.6     241.06667]]
Pred boxes after scaling (first 5):
[[ 796.       412.       886.       502.     ]
 [1092.       356.      1184.       435.     ]
 [ 668.       220.       745.       294.     ]
 [ 236.       527.       323.       598.     ]
 [ 937.00006  601.      1018.       678.     ]]

=== Image 434 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[  4.6      203.37778   30.4      243.91112 ]
 [101.6      146.48889  117.4      174.57779 ]
 [ 51.4      103.82223   75.6      144.35556 ]
 [ 38.600002  71.46667   60.8      107.022224]
 [ 63.2      171.02223   82.8      204.0889  ]]
Pred boxes after scaling (first 5):
[[ 23.      572.      152.      686.     ]
 [508.      412.      587.      491.00003]
 [257.      292.      378.      406.     ]
 [193.00002 201.      304.      301.     ]
 [316.      481.00003 414.      574.     ]]

=== Image 435 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[ 55.2       50.844444  77.8       86.755554]
 [102.6      147.55556  120.4      181.33334 ]
 [151.2       91.37778  169.2      122.31111 ]
 [ 77.6       55.466667  93.        81.77778 ]
 [113.4      111.288895 135.40001  151.82222 ]]
Pred boxes after scaling (first 5):
[[276.      143.      389.      244.     ]
 [513.      415.      602.      510.00003]
 [756.      257.      846.      344.     ]
 [388.      156.      465.      230.     ]
 [567.      313.00003 677.00006 427.     ]]

=== Image 436 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[186.40001  187.37778  209.8      227.55556 ]
 [  7.8      111.288895  30.6      145.06667 ]
 [201.8      130.48889  224.       165.33334 ]
 [ 89.4      223.2889   105.8      255.64445 ]
 [ 63.8       83.91111   85.       120.53333 ]]
Pred boxes after scaling (first 5):
[[ 932.00006  527.      1049.       640.     ]
 [  39.       313.00003  153.       408.     ]
 [1009.       367.      1120.       465.00003]
 [ 447.       628.       529.       719.     ]
 [ 319.       236.       425.       339.     ]]

=== Image 437 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[120.200005 169.24445  139.2      201.24445 ]
 [205.6      203.37778  226.2      238.22223 ]
 [ 30.6       53.68889   54.4       87.46667 ]
 [100.        15.644444 118.        48.355556]
 [ 67.4      206.22223   85.200005 239.2889  ]]
Pred boxes after scaling (first 5):
[[ 601.       476.       696.       566.     ]
 [1028.       572.      1131.       670.     ]
 [ 153.       151.       272.       246.     ]
 [ 500.        44.       590.       136.     ]
 [ 337.       580.       426.00003  673.     ]]

=== Image 438 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[ 24.2      183.46667   43.8      216.53334 ]
 [112.8      150.40001  130.6      183.82222 ]
 [ 75.4       23.111113  91.8       52.266666]
 [106.4       56.533333 121.8       87.46667 ]
 [198.6      136.88889  219.       170.66667 ]]
Pred boxes after scaling (first 5):
[[ 121.       516.       219.       609.     ]
 [ 564.       423.00003  653.       517.     ]
 [ 377.        65.00001  459.       147.     ]
 [ 532.       159.       609.       246.     ]
 [ 993.       385.      1095.       480.     ]]

=== Image 439 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[137.40001  210.84445  154.8      244.62222 ]
 [184.8       73.6      204.40001  106.66667 ]
 [118.6      192.       134.       220.0889  ]
 [232.8      193.77779  255.8      228.62222 ]
 [197.6        0.       216.        23.111113]]
Pred boxes after scaling (first 5):
[[ 687.00006  593.       774.       688.     ]
 [ 924.       207.      1022.00006  300.     ]
 [ 593.       540.       670.       619.     ]
 [1164.       545.      1279.       643.     ]
 [ 988.         0.      1080.        65.00001]]

=== Image 440 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[ 27.      103.82223  49.8     139.73334]
 [ 28.6      71.46667  50.4     104.53333]
 [181.6     183.46667 197.8     210.84445]
 [156.6      65.77778 174.40001  99.55556]
 [ 59.4     169.24445  77.4     199.46667]]
Pred boxes after scaling (first 5):
[[135.      292.      249.      393.     ]
 [143.      201.      252.      294.     ]
 [908.      516.      989.      593.     ]
 [783.      185.      872.00006 280.     ]
 [297.      476.      387.      561.     ]]

=== Image 441 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[230.6      207.2889   252.40001  241.06667 ]
 [117.6      145.42223  139.40001  187.02223 ]
 [162.40001  160.71112  182.8      197.33334 ]
 [ 41.4      151.11111   62.600002 184.8889  ]
 [ 33.4      118.04445   56.2      154.66667 ]]
Pred boxes after scaling (first 5):
[[1153.       583.      1262.       678.     ]
 [ 588.       409.       697.00006  526.     ]
 [ 812.00006  452.00003  914.       555.     ]
 [ 207.       425.       313.       520.00006]
 [ 167.       332.       281.       435.     ]]

=== Image 442 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[131.       155.02223  148.6      189.86667 ]
 [  0.       125.511116  17.2      154.66667 ]
 [231.2       13.866667 248.6       40.17778 ]
 [156.6      133.33334  175.2      164.26668 ]
 [ 10.400001  68.62222   37.8      108.08889 ]]
Pred boxes after scaling (first 5):
[[ 655.        436.00003   743.        534.      ]
 [   0.        353.         86.        435.      ]
 [1156.         39.       1243.        113.00001 ]
 [ 783.        375.00003   876.        462.00003 ]
 [  52.000004  193.        189.        304.      ]]

=== Image 443 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[ 63.2     126.57778  81.6     161.42223]
 [185.8     108.44445 206.40001 137.6    ]
 [123.        0.      140.40001  34.48889]
 [ 49.8     174.93333  65.      203.02223]
 [ 14.2     225.06667  34.8     255.64445]]
Pred boxes after scaling (first 5):
[[ 316.       356.       408.       454.     ]
 [ 929.       305.00003 1032.       387.00003]
 [ 615.         0.       702.00006   97.00001]
 [ 249.       492.       325.       571.     ]
 [  71.       633.       174.       719.     ]]

=== Image 444 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[183.2      123.73334  208.2      164.26668 ]
 [ 43.4      131.2       64.6      162.13333 ]
 [  0.       160.71112   18.800001 192.71112 ]
 [ 14.2      183.46667   37.600002 221.15556 ]
 [ 86.6      161.77779  102.4      189.86667 ]]
Pred boxes after scaling (first 5):
[[ 916.       348.      1041.       462.00003]
 [ 217.       369.       323.       456.     ]
 [   0.       452.00003   94.00001  542.     ]
 [  71.       516.       188.00002  622.     ]
 [ 433.       455.00003  512.       534.     ]]

=== Image 445 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[1.88600006e+02 1.25511116e+02 2.06000000e+02 1.56444443e+02]
 [2.34400009e+02 1.76000000e+02 2.53000000e+02 2.04088898e+02]
 [2.00000003e-01 1.67466675e+02 1.60000000e+01 2.06222229e+02]
 [2.25400009e+02 3.55555573e+01 2.48000000e+02 6.86222229e+01]
 [2.96000004e+01 2.27911118e+02 5.17999992e+01 2.55644455e+02]]
Pred boxes after scaling (first 5):
[[9.4300000e+02 3.5300000e+02 1.0300000e+03 4.4000000e+02]
 [1.1720000e+03 4.9500000e+02 1.2650000e+03 5.7400000e+02]
 [1.0000000e+00 4.7100003e+02 8.0000000e+01 5.8000000e+02]
 [1.1270000e+03 1.0000001e+02 1.2400000e+03 1.9300000e+02]
 [1.4800000e+02 6.4100000e+02 2.5900000e+02 7.1900000e+02]]

=== Image 446 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[152.8       81.066666 173.40001  117.68889 ]
 [131.40001  135.11111  146.8      164.26668 ]
 [175.2       93.511116 190.6      121.600006]
 [  9.400001 180.62222   25.800001 204.0889  ]
 [ 69.6       56.533333  85.        81.77778 ]]
Pred boxes after scaling (first 5):
[[764.       228.       867.00006  331.      ]
 [657.00006  380.       734.       462.00003 ]
 [876.       263.       953.       342.00003 ]
 [ 47.000004 508.       129.       574.      ]
 [348.       159.       425.       230.      ]]

=== Image 447 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[208.8       93.511116 231.40001  128.35556 ]
 [ 80.200005  68.62222   98.8       99.200005]
 [ 97.4      137.95557  114.4      169.95557 ]
 [222.6      142.57777  242.2      171.73334 ]
 [140.       145.42223  160.6      178.48889 ]]
Pred boxes after scaling (first 5):
[[1044.       263.      1157.       361.     ]
 [ 401.00003  193.       494.       279.     ]
 [ 487.       388.00003  572.       478.00003]
 [1113.       401.      1211.       483.     ]
 [ 700.       409.       803.       502.     ]]

=== Image 448 ===
Original size: (1280, 720), Resized: (256, 256)
Scale factors: x=5.0000, y=2.8125
Pred boxes before scaling (first 5):
[[173.6       99.200005 194.8      134.75555 ]
 [134.2      176.       152.8      210.84445 ]
 [231.2      169.24445  248.6      196.62222 ]
 [105.4       62.933334 130.2      102.4     ]
 [ 48.8      178.84445   69.4      216.53334 ]]
Pred boxes after scaling (first 5):
[[ 868.       279.       974.       379.     ]
 [ 671.       495.       764.       593.     ]
 [1156.       476.      1243.       553.     ]
 [ 527.       177.       651.       288.     ]
 [ 244.       503.00003  347.       609.     ]]

Loading and preparing COCO results...
Loading and preparing results...
DONE (t=0.07s)
creating index...
index created!
Running per-image evaluation...
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1.02s).
Accumulating evaluation results...
DONE (t=0.30s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.001
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.027
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.036
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
entering training


