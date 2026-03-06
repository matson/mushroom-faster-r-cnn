import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from PIL import Image

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
    plt.show()






