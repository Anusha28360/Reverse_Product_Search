import faiss
import numpy as np
import os

# Load image embeddings
embeddings = np.load("output/image_embeddings.npy")

# Convert to float32
embeddings = embeddings.astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

# Add embeddings
index.add(embeddings)


# Create models folder
os.makedirs("models", exist_ok=True)

# Save index
faiss.write_index(
    index,
    "models/faiss_index.bin"
)

print("\n✅ FAISS Index Created Successfully!")
print("Total Images Indexed:", index.ntotal)