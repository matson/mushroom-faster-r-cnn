
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils import load_checkpoint

# -------- LOAD MODEL --------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

load_checkpoint("best_fasterrcnn_mushroom_FULL.pth", model, device=device)
model.eval()

transform = T.Compose([T.ToTensor()])

def detect_mushrooms(img: Image.Image):
    """
    img: PIL.Image input
    Returns: PIL.Image with predicted bounding boxes overlaid
    """
    img_tensor = transform(img).to(device)

    with torch.no_grad():
        preds = model([img_tensor])

    pred_boxes = preds[0]['boxes'].cpu().numpy()
    pred_scores = preds[0]['scores'].cpu().numpy()

    img_np = np.array(img).copy()

    plt.figure(figsize=(8, 8))
    plt.imshow(img_np)

    for box, score in zip(pred_boxes, pred_scores):
        if score < 0.5:
            continue
        x1, y1, x2, y2 = box
        plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      fill=False, color='red', linewidth=2))
        plt.text(x1, y1 - 4, f"{score:.2f}", color='red', fontsize=6)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig("temp_result.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    return Image.open("temp_result.png")

# --- Launch Gradio interface
iface = gr.Interface(
    fn=detect_mushrooms,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Mushroom Detector",
    description="Upload an image and see mushroom detections from the fine-tuned Faster R-CNN model"
)

iface.launch()
