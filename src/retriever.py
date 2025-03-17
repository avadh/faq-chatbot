import faiss
import json
from sentence_transformers import SentenceTransformer

# Load FAISS index
index = faiss.read_index("models/faq_index.faiss")

# Load FAQ dataset
with open("models/faq_data.json", "r") as f:
    faq_data = json.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_faq(query):
    """Retrieve the most relevant FAQ."""
    query_embedding = model.encode([query], normalize_embeddings=True)
    _, indices = index.search(query_embedding, 1)  # Top-1 match
    return faq_data[indices[0][0]]

# Test
if __name__ == "__main__":
    user_query = "What is COVID?"
    retrieved = retrieve_faq(user_query)
    print(f"🔍 Matched FAQ: {retrieved['question']}\n✅ Answer: {retrieved['answer']}")
