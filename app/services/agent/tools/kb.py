import pickle
import faiss
import torch

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

from livekit.agents import llm

from app.utils.timer import Timer

#TODO Pre call tasks
with open("dev_scripts/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

def embed(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

def get_text_from_indices(indices):
    """Return the text chunks for each FAISS result index."""
    result = []
    for idx in indices:
        if 0 <= idx < len(chunks):
            result.append(chunks[idx])
        else:
            result.append("[INVALID INDEX]")
    return result


with Timer("Load Index, Tokenizer and Embedding Model"):
    index = faiss.read_index("dev_scripts/faiss.index")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = ORTModelForFeatureExtraction.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        export=True
    )

@llm.function_tool
async def ask_knowledge_base(question: str):
    """Ultra-fast retrieval with streaming context"""
    with Timer("KB Tool Total:"):
        with Timer("Embed Query"):
            q_emb = embed(question)
        with Timer("FAISS Search"):
            dist, idx = index.search(q_emb, k=3)    # top 3 matches
        indices = idx[0]                        # array of indices
        matched_text = get_text_from_indices(indices)
        context = "\n".join(matched_text)
        return context

@llm.function_tool
async def get_current_time(input: str) -> str:
    """Get the current time."""
    from datetime import datetime
    return f"The current time is {datetime.now().strftime('%I:%M %p')}" 


