# save as check_install.py in your project root
import arxiv
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
import anthropic
from fastapi import FastAPI
import torch

print(f"arxiv: OK")
print(f"PyMuPDF: OK")
print(f"chromadb: OK")
print(f"sentence-transformers: OK")
print(f"rank-bm25: OK")
print(f"langchain-core: OK")
print(f"langgraph: OK")
print(f"anthropic: OK")
print(f"fastapi: OK")
print(f"torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")