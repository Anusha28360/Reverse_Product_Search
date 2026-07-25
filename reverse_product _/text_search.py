import open_clip
import torch
import faiss
import numpy as np
import pandas as pd


# Load CLIP model
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

model.eval()


# Load tokenizer
tokenizer = open_clip.get_tokenizer("ViT-B-32")


# Load FAISS index
index = faiss.read_index(
    "models/faiss_index.bin"
)


# Load dataset
df = pd.read_csv(
    "dataset/styles.csv",
    engine="python",
    on_bad_lines="skip"
)


# Load image IDs
image_ids = np.load(
    "output/image_ids.npy"
)


# User enters query
query = input("\nEnter your search query: ")


# Convert text into embedding
text = tokenizer([query])

with torch.no_grad():
    text_features = model.encode_text(text)


# Normalize
text_features = text_features / text_features.norm(
    dim=-1,
    keepdim=True
)


# Convert to numpy
text_embedding = text_features.cpu().numpy().astype("float32")


# Search top 5
distances, indices = index.search(
    text_embedding,
    5
)


print("\nSearch Query:", query)
print("\nTop 5 Similar Products:\n")


# Display products
for i in indices[0]:

    img_id = image_ids[i]

    product = df[
        df["id"] == img_id
    ]["productDisplayName"].values

    if len(product) > 0:
        print(product[0])