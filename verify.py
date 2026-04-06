import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
from utils import load_checkpoint, MushroomCOCODataset


# -------- SETUP --------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

val_dataset = MushroomCOCODataset(
    images_dir="/home/matson/M18K_dataset/M18KV2_extracted/M18KV2/valid/rgb",
    annotations_file="/home/matson/M18K_dataset/M18KV2_extracted/M18KV2/valid/annotations_coco.json",
    resize=(448, 448)
)

# -------- LOAD MODEL --------
weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

load_checkpoint("best_fasterrcnn_mushroom_FULL.pth", model, device=device)
model.eval()

# -------- INFERENCE --------
indices = random.sample(range(len(val_dataset)), 5)

fig, axes = plt.subplots(1, 5, figsize=(25, 6))

for ax, img_index in zip(axes, indices):
    with torch.no_grad():
        img, target = val_dataset[img_index]
        pred = model([img.to(device)])[0]

    img_np = img.permute(1, 2, 0).numpy()
    ax.imshow(img_np)

    for box, score in zip(pred['boxes'], pred['scores']):
        if score > 0.5:
            x1, y1, x2, y2 = box.cpu().numpy()
            ax.add_patch(Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, color='red', linewidth=2
            ))
            ax.text(x1, y1 - 4, f"{score:.2f}", color='red', fontsize=6)

    ax.axis('off')
    ax.set_title(f"idx {img_index} — {len(pred['boxes'])} dets", fontsize=8)

plt.tight_layout()
plt.savefig("prediction_examples.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved prediction_examples.png")
