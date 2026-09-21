import os
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import core components derived from your codebase[cite: 1]
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import numpy as np
import re
import json
import time
from huggingface_hub import InferenceClient
from langgraph.graph import END, START, StateGraph
import operator
from typing import Annotated
from typing_extensions import TypedDict

app = FastAPI(title="Agentic Hybrid RAG API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from source codebase[cite: 1]
LLM_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"
CHUNK_SIZE, CHUNK_OVERLAP = 1000, 150
FETCH_K, FINAL_K = 10, 5
MAX_RETRIES = 1

UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

# Helper functions[cite: 1]
def tokenize(text):
    return re.findall(r"\w+", text.lower())

def label(doc):
    page = doc.metadata.get("page")
    return f"{doc.metadata['filename']}, p. {page}" if page else doc.metadata["filename"]

# Library Class for Hybrid Search[cite: 1]
class Library:
    def __init__(self):
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True},
        )
        self.clear()

    def clear(self):
        self.store = InMemoryVectorStore(self.embeddings)
        self.chunks = []
        self.bm25 = None

    def add_file(self, path, filename):
        if filename.lower().endswith(".pdf"):
            pages = PyPDFLoader(str(path)).load()
        else:
            pages = [Document(page_content=Path(path).read_text(errors="ignore"), metadata={})]
        pages = [p for p in pages if p.page_content.strip()]
        if not pages:
            raise ValueError("No readable text found.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        new_chunks = splitter.split_documents(pages)
        for chunk in new_chunks:
            page = chunk.metadata.get("page")
            chunk.metadata = {
                "filename": filename,
                "page": int(page) + 1 if page is not None else None,
                "chunk_id": len(self.chunks),
            }
            self.chunks.append(chunk)

        for start in range(0, len(new_chunks), 64):
            self.store.add_documents(new_chunks[start : start + 64])
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in self.chunks])
        return {"filename": filename, "pages": len(pages), "chunks": len(new_chunks)}

    def search(self, query, k=FINAL_K):
        if not self.chunks:
            return []
        fetch_k = min(FETCH_K, len(self.chunks))
        vector_ranking = [d.metadata["chunk_id"] for d in self.store.similarity_search(query, k=fetch_k)]
        scores = self.bm25.get_scores(tokenize(query))
        keyword_ranking = [int(i) for i in np.argsort(scores)[::-1][:fetch_k] if scores[i] > 0]

        fused = {}
        for ranking in (vector_ranking, keyword_ranking):
            for position, chunk_id in enumerate(ranking):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (60 + position + 1)
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self.chunks[i] for i in best]

library = Library()

# LLM & Agent Setup[cite: 1]
def ask_llm(system, user):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is not set in the backend environment variables.")
    client = InferenceClient(token=token)
    for attempt in range(2):
        try:
            reply = client.chat_completion(
                model=LLM_REPO_ID,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=700,
                temperature=0.2,
            )
            return reply.choices[0].message.content.strip()
        except Exception:
            if attempt == 1:
                raise
            time.sleep(2)

GRADE_SYSTEM = 'You check whether text passages help answer a question. Return ONLY a JSON object like {"relevant": [1, 3]} with numbers of helpful passages, or {"relevant": []} if none help.'
REWRITE_SYSTEM = 'Turn the question into a short keyword-style search query that would find the answer inside a book. Include key terms and synonyms. Output ONLY the query.'
GENERATE_SYSTEM = 'You answer questions about the user. Use ONLY the context below. If it does not contain the answer, say so plainly. Cite where facts come from, like (book.pdf, p. 12).'
NOT_FOUND = "I couldn't find anything about that in your document."

class State(TypedDict, total=False):
    question: str
    query: str
    docs: list
    relevant: list
    retries: int
    answer: str
    sources: list
    steps: Annotated[list, operator.add]

def retrieve(state):
    docs = library.search(state.get("query") or state["question"])
    return {"docs": docs, "steps": [f"Hybrid search found {len(docs)} passages"]}

def parse_relevant(raw, count):
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match: return None
    try:
        items = json.loads(match.group(0)).get("relevant")
    except: return None
    if not isinstance(items, list): return None
    return [i for i in items if isinstance(i, int) and 1 <= i <= count]

def grade(state):
    docs = state["docs"]
    if not docs: return {"relevant": [], "steps": ["Nothing to check"]}
    numbered = "\n\n".join(f"[{i}] {d.page_content[:500]}" for i, d in enumerate(docs, 1))
    try:
        raw = ask_llm(GRADE_SYSTEM, f"Question: {state['question']}\n\nPassages:\n{numbered}")
    except: raw = ""
    keep = parse_relevant(raw, len(docs))
    if keep is None:
        return {"relevant": docs, "steps": ["Relevance check unclear, keeping all"]}
    relevant = [docs[i - 1] for i in dict.fromkeys(keep)]
    return {"relevant": relevant, "steps": [f"Relevance check kept {len(relevant)} passages"]}

def rewrite_query(state):
    try:
        lines = ask_llm(REWRITE_SYSTEM, state["question"]).splitlines()
        query = lines[0].strip().strip('"') if lines else ""
    except: query = ""
    query = query or state["question"]
    return {"query": query, "retries": state.get("retries", 0) + 1, "steps": [f'Retrying with query: "{query}"']}

def generate(state):
    docs = state.get("relevant") or []
    used_fallback = not docs
    if used_fallback: docs = state.get("docs", [])[:3]
    if not docs: return {"answer": NOT_FOUND, "sources": [], "steps": ["No matching passages"]}
    context = "\n\n".join(f"[{i}] ({label(d)})\n{d.page_content}" for i, d in enumerate(docs, 1))
    answer = ask_llm(GENERATE_SYSTEM, f"Context:\n{context}\n\nQuestion: {state['question']}\n\nAnswer:")
    sources = [] if used_fallback else list(dict.fromkeys(label(d) for d in docs))
    return {"answer": answer, "sources": sources, "steps": ["Generated answer from passages"]}

def after_grade(state):
    if state.get("relevant"): return "generate"
    if state.get("docs") and state.get("retries", 0) < MAX_RETRIES: return "rewrite_query"
    return "generate"

graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("grade", grade)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("generate", generate)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges("grade", after_grade, {"generate": "generate", "rewrite_query": "rewrite_query"})
graph.add_edge("rewrite_query", "retrieve")
graph.add_edge("generate", END)
agent = graph.compile()

# API Endpoints
class QueryRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        info = library.add_file(file_path, file.filename)
        return {"message": "Success", "info": info}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/ask")
async def ask_question(payload: QueryRequest):
    if not library.chunks:
        raise HTTPException(status_code=400, detail="Please upload a document first.")
    try:
        result = agent.invoke({"question": payload.question, "retries": 0, "steps": []})
        return {
            "answer": result.get("answer"),
            "sources": result.get("sources", []),
            "steps": result.get("steps", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))