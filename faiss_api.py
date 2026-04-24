import os
import re
import pickle
import faiss
import numpy as np
from starlette.middleware.base import BaseHTTPMiddleware
import logging
from datetime import datetime, timedelta, timezone
import time

from dotenv import load_dotenv
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List

from sentence_transformers import SentenceTransformer

load_dotenv(override=True)
IST = timezone(timedelta(hours=5, minutes=30))

class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, IST)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

formatter = ISTFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

app = FastAPI()

# ==============================
# Config
# ==============================

MONGO_URI = os.environ["MONGO_URI"]
DB_NAME = "itsbot-db"
OUTPUT_DIR = "app/knowledge_base"

model = SentenceTransformer("all-MiniLM-L6-v2")

mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client[DB_NAME]

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()

        response = await call_next(request)

        duration = time.time() - start

        logger.info(f"{request.method} {request.url.path} took {duration:.3f}s")

        return response

class LogRequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        body = await request.body()

        logger.debug(f"Request URL: {request.url}")
        logger.debug(f"Headers: {dict(request.headers)}")
        logger.debug(f"Body: {body.decode('utf-8', errors='ignore')}")

        response = await call_next(request)
        return response

app.add_middleware(LogRequestMiddleware)
app.add_middleware(TimingMiddleware)

# ==============================
# Payload Schema
# ==============================

class KBUpsertPayload(BaseModel):

    text: str = ""
    dataToTrain: List[str] = []
    frequentlyAskedQuestions: List[str] = []
    removedfiles: List[str] = []

    namespace: str

    voiceText: str = ""

    files: List[str] = []
    imagesData: List[str] = []
    audioData: List[str] = []

    selectAll: bool = False
    selectAllDoc: bool = False
    selectAllImage: bool = False
    selectAllAudio: bool = False

    unSelected: List[str] = []
    unSelectedDoc: List[str] = []
    unSelectedImage: List[str] = []
    unSelectedAudio: List[str] = []

# ==============================
# Text Cleaning
# ==============================

def clean_text(text: str):

    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"#+\s*", "", text)

    text = text.replace("*", "").replace("_", "")
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()

# ==============================
# Chunking
# ==============================

def chunk_text(text, max_words=300):

    words = text.split()

    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

# ==============================
# Embedding
# ==============================

def embed_texts(texts):

    return model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

# ==============================
# FAISS Loader
# ==============================

def load_or_create_index(namespace, dim):

    index_path = f"{OUTPUT_DIR}/{namespace}_faiss.index"

    if os.path.exists(index_path):
        return faiss.read_index(index_path)

    return faiss.IndexFlatL2(dim)

# ==============================
# Fetch Helpers
# ==============================

async def fetch_by_ids(collection, ids, field):

    texts = []

    cursor = db[collection].find(
        {"_id": {"$in": [ObjectId(i) for i in ids]}},
        {field: 1}
    )

    async for doc in cursor:

        val = doc.get(field)

        if isinstance(val, list):
            texts.extend(val)

        elif isinstance(val, str):
            texts.append(val)

    return texts

# ==============================
# Resolve Select All
# ==============================

async def fetch_select_all(collection, kb_id, exclude_ids, field):

    texts = []

    query = {"knowledgeBase": ObjectId(kb_id)}

    if collection=="trainedImages" or collection=="trainedDocs":
        query = {"knowledgeBase": ObjectId(kb_id),
                 "isExtracted": True}
        
    if exclude_ids:
        query["_id"] = {"$nin": [ObjectId(i) for i in exclude_ids]}

    cursor = db[collection].find(query, {field: 1})

    async for doc in cursor:

        val = doc.get(field)

        if isinstance(val, list):
            texts.extend(val)

        elif isinstance(val, str):
            texts.append(val)

    return texts

# ==============================
# API
# ==============================
@app.post("/kb/upsert/{knowledgeBaseId}")
async def upsert_kb(knowledgeBaseId: str, payload: KBUpsertPayload):

    kb_id = ObjectId(knowledgeBaseId)
    namespace = payload.namespace

    all_texts = []
    faq_chunks = []

    # ---------------------------
    # Knowledgebase text
    # ---------------------------

    if payload.text:
        all_texts.append(payload.text)

    if payload.voiceText:
        all_texts.append(payload.voiceText)

    # ---------------------------
    # trainedlinks
    # ---------------------------

    if payload.selectAll:

        texts = await fetch_select_all(
            "trainedlinks",
            kb_id,
            payload.unSelected,
            "data"
        )

        all_texts.extend(texts)

    else:

        texts = await fetch_by_ids(
            "trainedlinks",
            payload.dataToTrain,
            "data"
        )

        all_texts.extend(texts)

    # ---------------------------
    # traineddocs
    # ---------------------------

    if payload.selectAllDoc:

        texts = await fetch_select_all(
            "traineddocs",
            kb_id,
            payload.unSelectedDoc,
            "data"
        )

        all_texts.extend(texts)

    else:

        texts = await fetch_by_ids(
            "traineddocs",
            payload.files,
            "data"
        )

        all_texts.extend(texts)
    # ---------------------------
    # trainedimages
    # ---------------------------

    if payload.selectAllImage:

        texts = await fetch_select_all(
            "trainedimages",
            kb_id,
            payload.unSelectedImage,
            "data"
        )

        all_texts.extend(texts)

    else:

        texts = await fetch_by_ids(
            "trainedimages",
            payload.imagesData,
            "data"
        )

        all_texts.extend(texts)

    # ---------------------------
    # trainedaudios
    # ---------------------------

    if payload.selectAllAudio:

        texts = await fetch_select_all(
            "trainedaudios",
            kb_id,
            payload.unSelectedAudio,
            "data"
        )

        all_texts.extend(texts)

    else:

        texts = await fetch_by_ids(
            "trainedaudios",
            payload.audioData,
            "data"
        )

        all_texts.extend(texts)

    # ---------------------------
    # FAQs (NO CHUNKING)
    # ---------------------------

    if payload.frequentlyAskedQuestions:

        cursor = db["faqs"].find(
            {
                "_id": {
                    "$in": [
                        ObjectId(i)
                        for i in payload.frequentlyAskedQuestions
                    ]
                }
            },
            {"question": 1, "answer": 1}
        )

        async for doc in cursor:

            q = doc.get("question")
            a = doc.get("answer")

            if q and a:
                faq_chunks.append(
                    f"Question: {q}\nAnswer: {a}"
                )

    # ---------------------------
    # Clean + Chunk
    # ---------------------------

    chunks = []

    for text in all_texts:

        cleaned = clean_text(text)

        chunks.extend(chunk_text(cleaned))

    chunks.extend(faq_chunks)

    if not chunks:
        return {"status": "no_chunks"}

    # ---------------------------
    # Embed
    # ---------------------------

    embeddings = embed_texts(chunks)

    # ---------------------------
    # FAISS Load/Create
    # ---------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    index = load_or_create_index(kb_id, embeddings.shape[1])

    index.add(embeddings)

    index_path = f"{OUTPUT_DIR}/{kb_id}_faiss.index"
    chunks_path = f"{OUTPUT_DIR}/{kb_id}_chunks.pkl"

    faiss.write_index(index, index_path)

    # ---------------------------
    # Update chunks
    # ---------------------------

    if os.path.exists(chunks_path):

        with open(chunks_path, "rb") as f:
            existing = pickle.load(f)

        existing.extend(chunks)

        chunks = existing

    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    return {
        "status": "success",
        "chunks_added": len(chunks),
        "knowledgeBaseId": str(kb_id),
        "namespace": namespace
    }


if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)