import open_clip
import torch
from PIL import Image

# Load CLIP model
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

model.eval()

# Image path
image_path = "dataset/images/15970.jpg"

# Load image
image = Image.open(image_path).convert("RGB")

# Preprocess image
image_input = preprocess(image).unsqueeze(0)

# Generate embedding
with torch.no_grad():
    image_features = model.encode_image(image_input)

print("Embedding Shape:", image_features.shape)
print("\nFirst 10 values:")
print(image_features[0][:10])