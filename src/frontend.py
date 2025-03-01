import gradio as gr
import threading
import uvicorn
from api import app

def start_fastapi():
    """Runs FastAPI backend in a separate thread."""
    uvicorn.run(app, host="0.0.0.0", port=5000)  #  Change FastAPI port to 5000 to avoid conflict with LLM

# Start FastAPI in a separate thread
threading.Thread(target=start_fastapi, daemon=True).start()

# Define chatbot frontend
def chat_with_bot(user_query):
    import requests
    response = requests.post("http://127.0.0.1:5000/query", json={"query": user_query}).json()
    return response["response"]

# Launch Gradio UI
gr.Interface(fn=chat_with_bot, inputs="text", outputs="text").launch(server_name="0.0.0.0", server_port=7860)
