import faiss
import pickle
import time
from sentence_transformers import SentenceTransformer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import torch

class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dur = time.perf_counter() - self.start
        print(f"\nTIMER: {self.name} took {dur:.4f} seconds")


with Timer("Load Index, Tokenizer and Embedding Model"):
    index = faiss.read_index("faiss.index")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = ORTModelForFeatureExtraction.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        export=True
    )
# with Timer("Load FAISS index and Embedding Model"):
#     index = faiss.read_index("faiss.index")
#     model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text):
    with Timer("ONNX Embedding"):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

# Load FAISS index


# Load chunk texts
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
    
def get_text_from_indices(indices):
    """Return the text chunks for each FAISS result index."""
    result = []
    for idx in indices:
        if 0 <= idx < len(chunks):
            result.append(chunks[idx])
        else:
            result.append("[INVALID INDEX]")
    return result

with Timer("Embed Query"):
    # Approach 2 - using ONNX model
    query = "What is this website about?"
    q_emb = embed(query)
    # Approach 1
    # query = "What is this website about?"
    # q_emb = model.encode([query], convert_to_numpy=True)

with Timer("FAISS Search"):
    dist, idx = index.search(q_emb, k=3)    # top 3 matches
indices = idx[0]                        # array of indices

print("Indices:", indices)

matched_text = get_text_from_indices(indices)

for i, text in zip(indices, matched_text):
    print(f"\n--- MATCH {i} ---\n{text}\n")