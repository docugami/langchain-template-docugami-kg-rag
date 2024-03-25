# Reference: https://huggingface.co/models
from rerankers import Reranker
import torch

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": device},
)

RERANKER = Reranker("mixedbread-ai/mxbai-rerank-base-v1", verbose=0)
