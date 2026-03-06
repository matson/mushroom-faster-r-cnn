

import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

import torchvision
from torchvision import transforms as T


# --- Ensure model is in eval mode
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Transform input image to tensor
transform = T.Compose([T.ToTensor()])

def detect_mushrooms(img: Image.Image):
    """
    img: PIL.Image input
    Returns: PIL.Image with predicted bounding boxes and masks overlaid
    """
    img_tensor = transform(img).to(device)
    
    with torch.no_grad():
        preds = model([img_tensor])
    
    pred_boxes = preds[0]['boxes'].cpu().numpy()
    pred_masks = preds[0]['masks'].cpu().numpy()  # shape [N,1,H,W]

    # Convert input to numpy for plotting
    img_np = np.array(img).copy()

    plt.figure(figsize=(8,8))
    plt.imshow(img_np)

    # Draw masks (alpha blend)
    for mask in pred_masks:
        mask_np = mask[0]  # remove channel dim
        plt.imshow(mask_np, alpha=0.3, cmap='Reds')

    # Draw bounding boxes
    for box in pred_boxes:
        x1, y1, x2, y2 = box
        plt.gca().add_patch(Rectangle((x1,y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2))

    plt.axis('off')

    # Save figure to PIL Image for Gradio output
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
    description="Upload an image of mushrooms and see the Mask R-CNN predictions"
)

iface.launch()