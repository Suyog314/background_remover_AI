
# # model_utils.py

# import torch
# import torchvision.transforms as T
# from torchvision.models.detection import maskrcnn_resnet50_fpn
# from PIL import Image
# import numpy as np
# import os

# COCO_INSTANCE_CATEGORY_NAMES = [
#     '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#     'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
#     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
#     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
#     'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
#     'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
#     'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
#     'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
#     'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
#     'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
#     'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
#     'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#     'teddy bear', 'hair drier', 'toothbrush'
# ]

# # Load model once
# model = maskrcnn_resnet50_fpn(weights="DEFAULT")
# model.eval()

# def remove_background(image_path, output_path, score_threshold=0.5):
#     image = Image.open(image_path).convert("RGB")
#     transform = T.Compose([T.ToTensor()])
#     input_tensor = transform(image).unsqueeze(0)

#     with torch.no_grad():
#         output = model(input_tensor)[0]

#     img_np = np.array(image)
#     H, W = img_np.shape[:2]
#     final_mask = np.zeros((H, W), dtype=np.uint8)

#     detected_labels = []
#     used_scores = []

#     for i in range(len(output["scores"])):
#         score = output["scores"][i].item()
#         if score >= score_threshold:
#             label_id = output["labels"][i].item()
#             label = COCO_INSTANCE_CATEGORY_NAMES[label_id]
#             mask = output["masks"][i, 0].numpy()

#             if np.sum(mask > 0.5) > 500:  # Skip very small detections
#                 final_mask[mask > 0.5] = 255
#                 detected_labels.append(label)
#                 used_scores.append(round(score, 2))

#     alpha = final_mask
#     rgba_image = np.dstack((img_np, alpha))

#     # Fallback: if mask is empty, return normal RGB image with solid alpha
#     if np.sum(alpha) == 0:
#         print("⚠️ No usable detections — returning original image.")
#         alpha[:, :] = 255
#         rgba_image = np.dstack((img_np, alpha))

#     Image.fromarray(rgba_image).save(output_path)

#     print("✅ Used detections:", list(zip(detected_labels, used_scores)))
#     print("🎯 Foreground pixels:", np.sum(alpha > 0))
#     return output_path



from rembg import remove
from PIL import Image
import numpy as np
import io

def remove_background(image_path, output_path):
    # Load input image
    with open(image_path, 'rb') as f:
        input_data = f.read()

    # Run U²-Net background removal
    output_data = remove(input_data)

    # Convert back to image
    result_image = Image.open(io.BytesIO(output_data)).convert("RGBA")

    # Save to file
    result_image.save(output_path)

    print("✅ Background removed and saved to:", output_path)
    return output_path
