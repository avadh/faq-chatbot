from fastapi import FastAPI
import requests
import numpy as np
from retriever import retrieve_faq
from retriever import model
from pydantic import BaseModel

# Update with your locally hosted LLM API
LLM_API_URL = "http://localhost:8000/v1/completions"

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def ask_question(request: QueryRequest):
    retrieved_faq = retrieve_faq(request.query)

    # Compute similarity score
    query_embedding = model.encode(request.query, normalize_embeddings=True)
    faq_embedding = model.encode(retrieved_faq["question"], normalize_embeddings=True)
    similarity_score = np.dot(query_embedding, faq_embedding)  # Cosine similarity
    print("Similarity Score:", similarity_score)

    # If similarity is below threshold, return "Out of scope"
    if similarity_score < 0.7:  # Adjust threshold as needed
        return {"response": "Out of scope for this FAQ chatbot."}

    context = f"FAQ: {retrieved_faq['question']} - {retrieved_faq['answer']}"
    print("Context:", context)
    # Generate response using LLM
    payload = {
        "model": "deepseek-r1:8b", # other model is llama3.2:latest
        "prompt": f"User Query: {request.query}\n\n{context}\n\nProvide a detailed response:",
        "temperature": 0.5
    }
    response = requests.post(LLM_API_URL, json=payload)

    # Print response details for debugging
    print("RAW RESPONSE STATUS CODE:", response.status_code)
    print("RAW RESPONSE TEXT:", response.text)

    try:
        response_json = response.json()
        generated_text = response_json.get("choices", [{}])[0].get("text", "No response generated.")
    except requests.exceptions.JSONDecodeError:
        generated_text = "Error: LLM API did not return valid JSON."

    return {"response": generated_text}

