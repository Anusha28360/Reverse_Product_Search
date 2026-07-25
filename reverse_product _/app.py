import streamlit as st
import open_clip
import torch
import faiss
import numpy as np
import pandas as pd
from PIL import Image
import os


# Page settings
st.set_page_config(
    page_title="AI Fashion Search",
    layout="wide"
)


st.title("🛍️ AI Fashion Search Engine")
st.write("Search fashion products using text")


# Load CLIP model
@st.cache_resource
def load_model():

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )

    model.eval()

    tokenizer = open_clip.get_tokenizer(
        "ViT-B-32"
    )

    return model, tokenizer


model, tokenizer = load_model()


# Load FAISS index
index = faiss.read_index(
    "models/faiss_index.bin"
)


# Load CSV
df = pd.read_csv(
    "dataset/styles.csv",
    engine="python",
    on_bad_lines="skip"
)


# Load image ids
image_ids = np.load(
    "output/image_ids.npy"
)


# Search box
query = st.text_input(
    "Enter product name:",
    "blue shirt"
)


if st.button("🔍 Search"):


    # Convert text to embedding
    text = tokenizer([query])


    with torch.no_grad():
        text_features = model.encode_text(text)


    # Normalize
    text_features = text_features / text_features.norm(
        dim=-1,
        keepdim=True
    )


    text_embedding = text_features.cpu().numpy().astype(
        "float32"
    )


    # Search top 5
    distances, indices = index.search(
        text_embedding,
        5
    )


    st.subheader("Top Similar Products")


    # Three cards per row
    cols = st.columns(3)


    count = 0


    for i in indices[0]:

        img_id = image_ids[i]


        product = df[
            df["id"] == img_id
        ]["productDisplayName"].values


        if len(product) > 0:


            image_path = f"dataset/images/{img_id}.jpg"


            with cols[count % 3]:

                st.write(
                    "**" + product[0] + "**"
                )


                if os.path.exists(image_path):

                    image = Image.open(
                        image_path
                    ).convert("RGB")


                    # Keep original ratio
                    image.thumbnail(
                        (450, 450)
                    )


                    st.image(
                        
                        image,
                        use_container_width=True
                    )


            count += 1