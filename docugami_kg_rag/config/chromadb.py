# ChromaDB
# Reference: https://python.langchain.com/docs/integrations/vectorstores/chroma
from pathlib import Path
from typing import List, Optional

from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings

from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.chroma import Chroma

import chromadb

CHROMA_DIRECTORY = Path("/tmp/docugami/chroma_db")
CHROMA_DIRECTORY.mkdir(parents=True, exist_ok=True)


def vector_store_index_exists(docset_id: str, embeddings: Embeddings) -> bool:
    persistent_client = chromadb.PersistentClient(path=str(CHROMA_DIRECTORY.absolute()))
    collections = persistent_client.list_collections()
    for c in collections:
        if c.name == docset_id:
            return True

    return False


def get_vector_store_index(docset_id: str, embeddings: Embeddings) -> Optional[VectorStore]:
    if vector_store_index_exists(docset_id, embeddings):
        return Chroma(
            collection_name=docset_id,
            persist_directory=str(CHROMA_DIRECTORY.absolute()),
            embedding_function=embeddings,
        )

    return None


def init_vector_store_index(docset_id: str, docs: List[Document], embeddings: Embeddings, force=True) -> VectorStore:
    if force and vector_store_index_exists(docset_id, embeddings):
        del_vector_store_index(docset_id)

    return Chroma.from_documents(
        documents=docs,
        collection_name=docset_id,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIRECTORY.absolute()),
    )


def del_vector_store_index(docset_id: str):
    persistent_client = chromadb.PersistentClient(path=str(CHROMA_DIRECTORY.absolute()))
    persistent_client.delete_collection(docset_id)
