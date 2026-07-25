import numpy as np
import faiss
import os

# Load embeddings
embeddings = np.load("output/image_embeddings.npy")

print("Embeddings Shape:", embeddings.shape)

# Convert to float32
embeddings = embeddings.astype("float32")

# Get embedding dimension
dimension = embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatIP(dimension)

# Add embeddings
index.add(embeddings)

# Save index
os.makedirs("models", exist_ok=True)
faiss.write_index(index, "models/faiss_index.bin")

print("\n✅ FAISS Index Created Successfully!")
print("Total Images Indexed:", index.ntotal)