
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
img_index = 0  # change this to view different validation images

with torch.no_grad():
    img, target = val_dataset[img_index]
    pred = model([img.to(device)])[0]

img_np = img.permute(1, 2, 0).numpy()
plt.figure(figsize=(10, 10))
plt.imshow(img_np)

for box, score in zip(pred['boxes'], pred['scores']):
    if score > 0.5:
        x1, y1, x2, y2 = box.cpu().numpy()
        plt.gca().add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, color='red', linewidth=2
        ))
        plt.text(x1, y1 - 4, f"{score:.2f}", color='red', fontsize=6)

plt.axis('off')
plt.title(f"Val image index {img_index} — {len(pred['boxes'])} detections")
plt.savefig("prediction_example.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved prediction_example.png")
