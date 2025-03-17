import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("data/faq_dataset.csv")

# Load embedding model
"""This is a comment"""
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate question embeddings
faq_embeddings = model.encode(df["question"].tolist(), normalize_embeddings=True)

# Create FAISS index
dimension = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(faq_embeddings)

# Save FAISS index
faiss.write_index(index, "models/faq_index.faiss")

# Save FAQ data
df.to_json("models/faq_data.json", orient="records")

print("✅ FAQ embeddings stored successfully.")
