[text](<../../../Downloads/AI Fashion Search Engine using CLIP and FAISS.pdf>)

 Reverse Product Search using OpenCLIP and FAISS

 ##
 Overview

This project is an AI-powered Reverse Product Search system that enables users to search for fashion products using text queries. It uses the OpenCLIP model to generate semantic embeddings and FAISS for fast similarity search. A Streamlit web application provides an interactive interface to display the most relevant fashion products.

 ## Features

- 🔍 Text-to-image semantic search
- 🤖 OpenCLIP (ViT-B-32) for feature extraction
- ⚡ FAISS-based similarity search for fast retrieval
- 🖥️ Interactive Streamlit web application
- 👕 Displays the top matching fashion products


## Technologies Used

- Python
- OpenCLIP
- PyTorch
- FAISS
- Streamlit
- Pandas
- NumPy
- Pillow

 Project Structure

 ## Reverse_Product_Search/
│
├── app.py
├── generate_embeddings.py
├── create_index.py
├── image_embedding.py
├── text_search.py
├── clip_test.py
├── check_images.py
├── build_index.py
├── README.md
├── dataset/
├── models/
└── output/
```

 Dataset

This project uses the **Fashion Product Images (Small)** dataset from Kaggle.

**Dataset Link:**

https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

> **Note:** The dataset is not included in this repository because of GitHub file size limitations. Please download it separately from the above link.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Reverse_Product_Search.git
```

Move to the project folder:

```bash
cd Reverse_Product_Search
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Step 1: Generate Image Embeddings

```bash
python generate_embeddings.py
```

### Step 2: Create the FAISS Index

```bash
python create_index.py
```

### Step 3: Launch the Streamlit Application

```bash
streamlit run app.py
```

---

## 📷 Output

The application accepts a text query such as:

```
blue shirt
```

It retrieves and displays the most visually similar fashion products from the dataset.

---

##  Future Improvements

- Support image-to-image search
- Improve search accuracy using larger CLIP models
- Deploy the application on Streamlit Cloud
- Add product filters (category, colour, gender)

---

##  Author

**Anusha L**

- LinkedIn: https://linkedin.com/in/anusha-l-874100300
- GeeksforGeeks: https://www.geeksforgeeks.org/profile/anushaj421

---

##  License

This project is developed for educational and learning purposes.
