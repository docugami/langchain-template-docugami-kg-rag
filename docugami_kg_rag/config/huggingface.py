# Reference: https://huggingface.co/models
import torch
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from rerankers import Reranker

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": device},
)

RERANKER = Reranker("mixedbread-ai/mxbai-rerank-base-v1", verbose=0)
