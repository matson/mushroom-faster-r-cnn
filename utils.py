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

Running sanity check for mAP calculation...
creating index...
index created!
=== Debug Info BEFORE loadRes ===
GT first 5 annotations: [{'id': 0, 'image_id': 0, 'category_id': 1, 'bbox': [25, 191, 84, 74], 'area': 4755.375, 'segmentation': [[44, 191, 44, 193, 43, 194, 43, 194, 42, 195, 41, 195, 40, 196, 39, 196, 38, 197, 37, 197, 31, 204, 31, 205, 29, 206, 29, 207, 29, 207, 29, 208, 27, 209, 25, 209, 25, 234, 27, 234, 28, 235, 28, 235, 29, 236, 29, 237, 29, 237, 29, 238, 30, 239, 30, 239, 31, 241, 31, 241, 32, 242, 32, 243, 34, 245, 34, 245, 35, 247, 35, 247, 41, 253, 41, 253, 43, 255, 44, 255, 45, 256, 46, 256, 47, 257, 48, 257, 49, 258, 50, 258, 51, 259, 53, 259, 53, 260, 55, 260, 55, 261, 56, 261, 57, 261, 59, 261, 59, 262, 60, 262, 61, 263, 61, 265, 73, 265, 73, 263, 75, 261, 76, 261, 77, 260, 77, 259, 78, 258, 79, 258, 81, 255, 83, 255, 83, 255, 87, 255, 88, 254, 91, 254, 92, 253, 93, 253, 94, 253, 95, 253, 96, 251, 97, 251, 99, 249, 100, 249, 101, 248, 101, 248, 103, 247, 103, 247, 104, 246, 105, 246, 107, 243, 107, 243, 108, 242, 108, 241, 109, 240, 109, 239, 109, 238, 109, 229, 109, 228, 109, 224, 108, 223, 108, 222, 107, 221, 107, 220, 107, 219, 107, 219, 106, 218, 106, 217, 105, 216, 105, 215, 105, 214, 105, 213, 104, 213, 104, 212, 103, 211, 103, 211, 102, 209, 102, 209, 101, 207, 101, 207, 94, 200, 93, 200, 91, 198, 91, 198, 89, 196, 87, 196, 87, 195, 86, 195, 85, 194, 84, 194, 83, 193, 83, 193, 82, 193, 82, 191]], 'iscrowd': 0}, {'id': 1, 'image_id': 0, 'category_id': 1, 'bbox': [935, 276, 85, 71], 'area': 4588.0, 'segmentation': [[963, 276, 963, 278, 962, 279, 961, 279, 961, 279, 959, 279, 959, 280, 958, 280, 957, 281, 956, 281, 955, 281, 955, 281, 953, 283, 953, 283, 951, 284, 951, 284, 948, 287, 947, 287, 940, 294, 940, 295, 939, 296, 939, 297, 938, 297, 938, 298, 937, 299, 935, 299, 935, 324, 937, 324, 937, 325, 937, 325, 938, 326, 938, 327, 939, 328, 939, 329, 941, 331, 941, 331, 943, 333, 943, 333, 945, 336, 945, 337, 948, 339, 949, 339, 949, 340, 950, 340, 951, 341, 951, 341, 953, 342, 954, 342, 955, 343, 956, 343, 957, 343, 958, 343, 959, 344, 960, 344, 961, 345, 961, 345, 962, 345, 962, 347, 973, 347, 973, 345, 974, 344, 976, 344, 977, 343, 979, 343, 979, 343, 981, 343, 982, 342, 985, 342, 985, 341, 987, 341, 988, 341, 990, 341, 991, 340, 993, 340, 993, 339, 998, 339, 999, 339, 1001, 339, 1001, 338, 1003, 338, 1003, 337, 1004, 337, 1006, 335, 1007, 335, 1011, 331, 1011, 331, 1013, 329, 1013, 328, 1014, 327, 1014, 326, 1015, 325, 1015, 324, 1015, 323, 1015, 322, 1016, 321, 1016, 319, 1017, 318, 1019, 318, 1019, 306, 1017, 306, 1017, 305, 1017, 305, 1016, 304, 1016, 303, 1015, 302, 1015, 301, 1015, 301, 1015, 299, 1014, 299, 1014, 297, 1013, 296, 1013, 295, 1012, 295, 1012, 294, 1011, 293, 1011, 293, 1010, 291, 1010, 291, 1005, 286, 1005, 286, 1003, 284, 1002, 284, 1001, 283, 1001, 283, 999, 282, 999, 282, 998, 281, 997, 281, 997, 281, 996, 281, 995, 280, 995, 280, 994, 279, 993, 279, 992, 278, 992, 276]], 'iscrowd': 0}, {'id': 2, 'image_id': 0, 'category_id': 1, 'bbox': [951, 545, 101, 90], 'area': 6720.0, 'segmentation': [[975, 545, 975, 547, 973, 549, 973, 549, 972, 549, 971, 549, 971, 550, 970, 550, 969, 551, 968, 551, 967, 553, 966, 553, 958, 561, 958, 561, 957, 562, 957, 563, 955, 565, 955, 565, 954, 567, 954, 567, 953, 569, 951, 569, 951, 597, 953, 597, 954, 599, 954, 599, 955, 601, 955, 601, 956, 602, 956, 603, 957, 604, 957, 605, 958, 605, 958, 606, 959, 607, 959, 607, 961, 609, 961, 610, 971, 621, 972, 621, 974, 623, 975, 623, 975, 623, 976, 623, 977, 625, 978, 625, 979, 625, 979, 625, 980, 626, 981, 626, 981, 627, 982, 627, 983, 627, 983, 627, 984, 628, 985, 628, 985, 629, 987, 629, 987, 629, 988, 629, 989, 630, 990, 630, 991, 631, 991, 631, 992, 631, 994, 631, 995, 632, 997, 632, 998, 633, 999, 633, 999, 633, 999, 635, 1015, 635, 1015, 633, 1015, 633, 1017, 633, 1017, 632, 1019, 632, 1020, 631, 1021, 631, 1021, 631, 1023, 631, 1024, 630, 1025, 630, 1025, 629, 1026, 629, 1027, 629, 1027, 629, 1028, 628, 1029, 628, 1029, 627, 1030, 627, 1031, 626, 1032, 626, 1033, 625, 1034, 625, 1036, 623, 1037, 623, 1042, 617, 1042, 617, 1043, 616, 1043, 615, 1044, 614, 1044, 613, 1045, 613, 1045, 612, 1045, 611, 1045, 611, 1046, 610, 1046, 609, 1047, 608, 1047, 607, 1047, 606, 1047, 604, 1048, 603, 1048, 601, 1049, 601, 1049, 599, 1049, 598, 1051, 598, 1051, 593, 1049, 593, 1049, 593, 1049, 591, 1048, 590, 1048, 589, 1047, 588, 1047, 587, 1047, 586, 1047, 585, 1046, 584, 1046, 582, 1045, 581, 1045, 581, 1045, 580, 1045, 579, 1044, 579, 1044, 577, 1043, 577, 1043, 575, 1041, 573, 1041, 573, 1041, 572, 1041, 571, 1039, 569, 1039, 569, 1030, 560, 1029, 560, 1027, 558, 1027, 558, 1026, 557, 1025, 557, 1025, 557, 1024, 557, 1023, 556, 1023, 556, 1022, 555, 1021, 555, 1020, 554, 1019, 554, 1019, 553, 1018, 553, 1017, 553, 1017, 553, 1016, 552, 1015, 552, 1014, 551, 1013, 551, 1013, 550, 1012, 550, 1011, 549, 1011, 549, 1010, 549, 1009, 549, 1008, 547, 1008, 545]], 'iscrowd': 0}, {'id': 3, 'image_id': 0, 'category_id': 1, 'bbox': [185, 217, 79, 77], 'area': 4733.25, 'segmentation': [[216, 217, 216, 219, 215, 220, 215, 220, 214, 221, 213, 221, 212, 221, 211, 221, 210, 222, 209, 222, 209, 223, 207, 223, 207, 223, 206, 223, 205, 224, 205, 224, 204, 225, 203, 225, 203, 225, 201, 225, 199, 227, 199, 227, 197, 229, 196, 229, 194, 231, 194, 232, 192, 234, 192, 235, 190, 237, 190, 237, 189, 238, 189, 239, 189, 239, 189, 240, 187, 241, 185, 241, 185, 273, 188, 273, 189, 273, 189, 274, 189, 275, 189, 275, 190, 276, 190, 277, 192, 279, 192, 279, 197, 285, 198, 285, 199, 286, 200, 286, 201, 287, 202, 287, 203, 289, 204, 289, 205, 289, 205, 289, 206, 290, 207, 290, 207, 291, 209, 291, 209, 291, 210, 291, 211, 292, 211, 294, 233, 294, 233, 292, 233, 291, 234, 291, 235, 291, 236, 291, 237, 290, 238, 290, 239, 289, 240, 289, 241, 289, 241, 289, 243, 287, 243, 287, 244, 287, 245, 287, 251, 281, 251, 280, 252, 279, 252, 278, 253, 277, 253, 275, 254, 275, 254, 274, 255, 273, 255, 272, 256, 271, 256, 271, 257, 270, 257, 269, 257, 268, 257, 267, 258, 266, 258, 265, 259, 264, 259, 263, 259, 263, 259, 262, 260, 261, 260, 261, 261, 260, 261, 259, 262, 257, 262, 256, 263, 255, 263, 253, 263, 253, 263, 251, 264, 251, 264, 248, 265, 247, 265, 243, 264, 243, 264, 241, 263, 240, 263, 239, 263, 239, 263, 237, 262, 237, 262, 236, 260, 234, 260, 233, 255, 229, 255, 228, 255, 228, 251, 224, 250, 224, 249, 223, 247, 223, 247, 222, 246, 222, 245, 221, 245, 221, 244, 221, 243, 221, 242, 219, 242, 217]], 'iscrowd': 0}, {'id': 4, 'image_id': 0, 'category_id': 1, 'bbox': [801, 484, 87, 82], 'area': 5424.0, 'segmentation': [[827, 484, 827, 486, 827, 487, 826, 487, 825, 487, 824, 487, 823, 488, 823, 488, 822, 489, 821, 489, 821, 489, 820, 489, 819, 491, 818, 491, 817, 492, 816, 492, 811, 497, 811, 497, 809, 499, 809, 500, 808, 501, 808, 502, 807, 503, 807, 503, 807, 504, 807, 505, 806, 505, 806, 506, 805, 507, 805, 507, 805, 508, 805, 509, 803, 510, 801, 510, 801, 532, 803, 532, 805, 533, 805, 535, 805, 535, 805, 536, 806, 537, 806, 537, 807, 538, 807, 539, 808, 540, 808, 541, 809, 542, 809, 543, 811, 545, 811, 545, 817, 551, 818, 551, 822, 555, 823, 555, 823, 556, 824, 556, 825, 557, 825, 557, 826, 557, 827, 557, 827, 558, 828, 558, 829, 559, 829, 559, 830, 559, 831, 559, 832, 560, 833, 560, 833, 561, 835, 561, 836, 561, 839, 561, 839, 562, 844, 562, 845, 563, 848, 563, 849, 563, 851, 563, 851, 564, 851, 566, 855, 566, 855, 564, 855, 563, 856, 563, 857, 563, 859, 563, 859, 562, 861, 562, 861, 561, 863, 561, 864, 561, 865, 561, 867, 559, 868, 559, 869, 559, 869, 559, 870, 558, 871, 558, 872, 557, 873, 557, 875, 555, 875, 555, 879, 551, 879, 551, 881, 549, 881, 548, 882, 547, 882, 546, 883, 545, 883, 545, 883, 544, 883, 543, 884, 543, 884, 542, 885, 541, 885, 541, 885, 540, 885, 539, 887, 537, 889, 537, 889, 517, 887, 517, 886, 516, 886, 515, 885, 515, 885, 513, 885, 513, 885, 512, 884, 511, 884, 511, 883, 510, 883, 509, 883, 509, 883, 508, 882, 507, 882, 507, 881, 505, 881, 505, 880, 504, 880, 503, 879, 503, 879, 502, 878, 501, 878, 500, 875, 497, 875, 496, 872, 493, 871, 493, 869, 491, 868, 491, 867, 489, 866, 489, 865, 488, 864, 488, 863, 487, 862, 487, 861, 487, 861, 487, 860, 486, 860, 484]], 'iscrowd': 0}]
Pred first 5 results: [{'image_id': 0, 'category_id': 1, 'bbox': [5.0, 67.9111099243164, 16.80000114440918, 26.311111450195312], 'score': 1.0}, {'image_id': 0, 'category_id': 1, 'bbox': [187.0, 98.13333892822266, 17.0, 25.24443817138672], 'score': 1.0}, {'image_id': 0, 'category_id': 1, 'bbox': [190.1999969482422, 193.7777862548828, 20.20001220703125, 32.0], 'score': 1.0}, {'image_id': 0, 'category_id': 1, 'bbox': [37.0, 77.15555572509766, 15.799999237060547, 27.377777099609375], 'score': 1.0}, {'image_id': 0, 'category_id': 1, 'bbox': [160.1999969482422, 172.08889770507812, 17.400009155273438, 29.155548095703125], 'score': 1.0}]
GT image_ids: [0, 0, 0, 0, 0]
Pred image_ids: [0, 0, 0, 0, 0]
GT category_ids: [1, 1, 1, 1, 1]
Pred category_ids: [1, 1, 1, 1, 1]
GT bboxes: [[25, 191, 84, 74], [935, 276, 85, 71], [951, 545, 101, 90], [185, 217, 79, 77], [801, 484, 87, 82]]
Pred bboxes: [[5.0, 67.9111099243164, 16.80000114440918, 26.311111450195312], [187.0, 98.13333892822266, 17.0, 25.24443817138672], [190.1999969482422, 193.7777862548828, 20.20001220703125, 32.0], [37.0, 77.15555572509766, 15.799999237060547, 27.377777099609375], [160.1999969482422, 172.08889770507812, 17.400009155273438, 29.155548095703125]]
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.85s).
Accumulating evaluation results...
DONE (t=0.02s).


