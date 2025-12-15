import requests
import pickle
from bs4 import BeautifulSoup
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_md_text(file_path: str) -> str:
    """
    Load text from a Markdown file and return clean plain text.
    Removes links, images, code blocks, markdown symbols, etc.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        md = f.read()

    # Remove code blocks ```...```
    md = re.sub(r"```[\s\S]*?```", "", md)

    # Remove inline code `...`
    md = re.sub(r"`([^`]*)`", r"\1", md)

    # Remove images ![alt](url)
    md = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", md)

    # Remove links [text](url) -> keep the text
    md = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", md)

    # Remove markdown headings (##, ###)
    md = re.sub(r"#+\s*", "", md)

    # Remove bold/italics formatting markers (* _ ** __)
    md = md.replace("*", "").replace("_", "")

    # Collapse multiple newlines
    md = re.sub(r"\n{2,}", "\n", md)

    return md.strip()

def fetch_website_text(url: str) -> str:
    html = requests.get(url, timeout=20).text
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(" ", strip=True)


def chunk_text(text: str, max_words: int = 300):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])


def embed_texts(texts):
    return model.encode(texts, convert_to_numpy=True)


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


if __name__ == "__main__":
    url = "https://example.com"
    file = "data/eminence.md"
    text = load_md_text(file)
    # text = fetch_website_text(url)
    chunks = list(chunk_text(text))
    embeddings = embed_texts(chunks)
    index = build_faiss_index(embeddings)

    faiss.write_index(index, "faiss.index")

    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
