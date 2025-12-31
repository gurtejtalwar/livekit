import pickle
import faiss
import torch

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

from livekit import api
from livekit.agents import llm, get_job_context, RunContext

from app.utils.timer import Timer

class KnowledgeBase:
    def __init__(self, index, chunks):
        self.index = index
        self.chunks = chunks

    def search(self, query_emb, k=3):
        dist, idx = self.index.search(query_emb, k)
        indices = idx[0]
        return [
            self.chunks[i] if 0 <= i < len(self.chunks) else "[INVALID INDEX]"
            for i in indices
        ]

KB_CACHE = {} #TODO HAZARD use redis

#TODO Use resource centre id instead of agent id
def load_knowledge_base(resource_centre_id: str) -> KnowledgeBase: 
    if resource_centre_id in KB_CACHE:
        return KB_CACHE[resource_centre_id]

    with Timer(f"Load KB for {resource_centre_id}"):
        index = faiss.read_index(f"kbs/{resource_centre_id}/faiss.index")
        with open(f"kbs/{resource_centre_id}/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

    kb = KnowledgeBase(index=index, chunks=chunks)
    KB_CACHE[resource_centre_id] = kb
    return kb

def make_ask_knowledge_base_tool(kb: KnowledgeBase):

    @llm.function_tool
    async def ask_knowledge_base(question: str):
        with Timer("KB Tool Total"):
            with Timer("Embed Query"):
                q_emb = embed(question)

            with Timer("FAISS Search"):
                results = kb.search(q_emb, k=3)

            return "\n".join(results)

    return ask_knowledge_base

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

async def hangup_call(ctx: RunContext):
    # Ensure any pending agent speech is finished before killing the room
    await ctx.wait_for_playout()
    await api.room.delete_room(
        api.DeleteRoomRequest(room=ctx.room.name)
    )


@llm.function_tool
async def end_call(ctx: RunContext,
                   dummy: str = ""):
    """Use this tool when the user has signaled they wish to end the current call."""
    session = ctx.session
    session.generate_reply(instructions="You/User have chosen to end the call.")
    await ctx.wait_for_playout() # Ensure agent finishes speaking
    job_ctx = get_job_context()
    if job_ctx:
        # Use job_ctx.api to delete the room
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(room=job_ctx.room.name)
        )