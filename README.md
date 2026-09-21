# Agentic-Hybrid-RAG

# 🤖 Agentic Hybrid RAG Full-Stack Application

A full-stack Retrieval-Augmented Generation (RAG) web application built using **FastAPI** (Python) and **React** (JavaScript). This project allows users to upload PDF documents, index them using a hybrid search mechanism, and chat with an intelligent agent that answers questions with accurate page citations.

---

## 📖 About the Project

Standard RAG systems often retrieve irrelevant text chunks, leading to hallucinated answers. This project implements an **Agentic Hybrid RAG** architecture (adapted from) to solve this problem:
1. **Hybrid Search:** Combines **BM25 keyword search** (exact term matching) and **Vector semantic search** (meaning matching), merged together using **Reciprocal Rank Fusion (RRF)**.
2. **LangGraph Agent Loop:** Instead of blindly trusting search results, a LangGraph agent reviews/grades the retrieved passages. If the passages are irrelevant, the agent automatically rewrites the query and searches again before generating the final response.

---

## 🏗️ Architecture & How It Works

```text
[ React Frontend ] --(Upload PDF)--> [ FastAPI Backend ] 
                                           │
                                    (Splits & Chunks PDF)
                                           │
                                  ┌────────┴────────┐
                                  ▼                 ▼
                             [ BM25 Index ]   [ Vector Store ]
                                  │                 │
                                  └────────┬────────┘
                                           ▼
                               (Reciprocal Rank Fusion)
                                           │
                                           ▼
                              [ LangGraph Agent Loop ]
                             (Retrieve -> Grade -> Rewrite)
                                           │
                                           ▼
                                [ Qwen LLM Generation ]
                                           │
                                           ▼
[ React Frontend ] <---(Returns Answer & Sources)--┘
