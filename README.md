# featuring a ResNet-50 FPN backbone

This project fine-tunes a Faster R-CNN model with a ResNet-50 FPN backbone — pretrained on the COCO dataset — to detect mushrooms in images. The model is trained on
the M18K dataset, a large collection of mushroom images with COCO-style annotations containing bounding boxes and segmentation masks. The classification head was
replaced to output two classes: background and mushroom. Training runs for 10 epochs with SGD optimization and a step learning rate scheduler, saving checkpoints
after each epoch and tracking both training and validation loss curves. After each epoch, COCO-style mAP is evaluated on the validation set to measure detection     
quality, with AP@IoU=0.50 as the primary metric.
                                                                                                                                                                       
Data Preparation

The dataset's COCO annotations contained multiple category IDs (1 and 2) representing different mushroom types, but since the goal is simply detecting mushrooms as a
single class, all category IDs were remapped to 1 before evaluation so that COCOeval could correctly match predictions to ground truth. Additionally, due to the
memory constraints of training on a GeForce GTX 960M with only 4GB of VRAM, all images were resized from their original resolution down to 256×256 pixels. This
required scaling bounding box coordinates proportionally at load time, and then scaling them back to original image coordinates before passing predictions to        
COCOeval — since COCOeval always compares against the original annotation dimensions. A batch size of 1 was also used to stay within the 4GB memory limit.
