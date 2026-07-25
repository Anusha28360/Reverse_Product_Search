import os
import torch
import open_clip
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# Load CLIP model
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

model.eval()


# Load dataset
df = pd.read_csv(
    "dataset/styles.csv",
    engine="python",
    on_bad_lines="skip"
)


embeddings = []
image_ids = []


# Use first 5000 images
total_images = 5000

for _, row in tqdm(df.head(total_images).iterrows(), total=total_images):

    image_path = f"dataset/images/{row['id']}.jpg"

    if os.path.exists(image_path):

        try:
            image = Image.open(image_path).convert("RGB")

            image_input = preprocess(image).unsqueeze(0)

            with torch.no_grad():
                feature = model.encode_image(image_input)

            # Normalize
            feature = feature / feature.norm(dim=-1, keepdim=True)

            embeddings.append(
                feature.squeeze().numpy()
            )

            image_ids.append(row["id"])


        except Exception as e:
            continue


# Convert to numpy
embeddings = np.array(embeddings).astype("float32")
image_ids = np.array(image_ids)


# Create output folder
os.makedirs("output", exist_ok=True)


# Save files
np.save(
    "output/image_embeddings.npy",
    embeddings
)

np.save(
    "output/image_ids.npy",
    image_ids
)


print("\n✅ Embeddings Saved Successfully!")
print("Embeddings Shape:", embeddings.shape)
print("Total Images Processed:", len(image_ids))