import hashlib
import os
from pathlib import Path
import pickle
from typing import Dict, List

from langchain.schema import Document
from langchain.storage.in_memory import InMemoryStore

from langchain_docugami.document_loaders.docugami import DocugamiLoader
from langchain_docugami.retrievers.fused_summary import FULL_DOC_SUMMARY_ID_KEY, SOURCE_KEY
from langchain_docugami.tools.retrieval import (
    chunks_to_direct_retriever_tool_description,
    docset_name_to_direct_retriever_tool_function_name,
)

from docugami_kg_rag.config import (
    INCLUDE_XML_TAGS,
    INDEXING_LOCAL_STATE_PATH,
    MAX_CHUNK_TEXT_LENGTH,
    MIN_CHUNK_TEXT_LENGTH,
    PARENT_HIERARCHY_LEVELS,
    SUB_CHUNK_TABLES,
    EMBEDDINGS,
    SMALL_CONTEXT_INSTRUCT_LLM,
    get_vector_store_index,
    init_vector_store_index,
    del_vector_store_index,
)
from docugami_kg_rag.helpers.documents import build_full_doc_summary_mappings, build_chunk_summary_mappings
from docugami_kg_rag.helpers.reports import ReportDetails, build_report_details
from docugami_kg_rag.helpers.retrieval import LocalIndexState


def read_all_local_index_state() -> Dict[str, LocalIndexState]:
    if not Path(INDEXING_LOCAL_STATE_PATH).is_file():
        return {}  # not found

    with open(INDEXING_LOCAL_STATE_PATH, "rb") as file:
        return pickle.load(file)


def update_local_index(
    docset_id: str,
    full_doc_summaries_by_id: Dict[str, Document],
    chunks_by_id: Dict[str, Document],
    direct_tool_function_name: str,
    direct_tool_description: str,
    report_details: List[ReportDetails],
):
    """
    Read and update local index
    """

    state = read_all_local_index_state()

    full_doc_summaries_by_id_store = InMemoryStore()
    full_doc_summaries_by_id_store.mset(list(full_doc_summaries_by_id.items()))

    chunks_by_id_store = InMemoryStore()
    chunks_by_id_store.mset(list(chunks_by_id.items()))

    doc_index_state = LocalIndexState(
        full_doc_summaries_by_id=full_doc_summaries_by_id_store,
        chunks_by_id=chunks_by_id_store,
        retrieval_tool_function_name=direct_tool_function_name,
        retrieval_tool_description=direct_tool_description,
        reports=report_details,
    )
    state[docset_id] = doc_index_state

    # Serialize state to disk (Deserialized in chain)
    store_local_path = Path(INDEXING_LOCAL_STATE_PATH)
    os.makedirs(os.path.dirname(store_local_path), exist_ok=True)
    with open(store_local_path, "wb") as file:
        pickle.dump(state, file)


def populate_vector_index(docset_id: str, chunks: List[Document], overwrite=False):
    """
    Create index if it does not exist, delete and overwrite if overwrite is specified.
    """

    vector_store = get_vector_store_index(docset_id, EMBEDDINGS)

    if vector_store is not None:
        print(f"Vector store index already exists for {docset_id}.")
        if overwrite is True:
            print(f"Overwrite is {overwrite}, existing index will be deleted and re-created")
        else:
            print(f"Overwrite is {overwrite}, will just reuse existing index (any new docs will not be added)")
            return

    print(f"Embedding documents into vector store for {docset_id}...")

    vector_store = init_vector_store_index(docset_id, chunks, EMBEDDINGS, overwrite)

    print(f"Done embedding documents into vector store for {docset_id}")


def index_docset(docset_id: str, name: str, overwrite=False):
    """
    Indexes the given docset
    """

    print(f"Indexing {name} (ID: {docset_id})")

    loader = DocugamiLoader(
        docset_id=docset_id,
        file_paths=None,
        document_ids=None,
        min_text_length=MIN_CHUNK_TEXT_LENGTH,
        max_text_length=MAX_CHUNK_TEXT_LENGTH,  # type: ignore
        sub_chunk_tables=SUB_CHUNK_TABLES,
        include_xml_tags=INCLUDE_XML_TAGS,
        parent_hierarchy_levels=PARENT_HIERARCHY_LEVELS,
        include_project_metadata_in_doc_metadata=False,  # not used, so lighten the vector index
    )

    chunks = loader.load()

    # Build separate maps of chunks, and parents
    parent_chunks_by_id: Dict[str, Document] = {}
    chunks_by_source: Dict[str, List[str]] = {}
    for chunk in chunks:
        chunk_id = str(chunk.metadata.get("id"))
        chunk_source = str(chunk.metadata.get(SOURCE_KEY))
        parent_chunk_id = chunk.metadata.get(loader.parent_id_key)
        if not parent_chunk_id:
            # parent chunk, we will use this (for expanded context) as our chunk
            parent_chunks_by_id[chunk_id] = chunk
        else:
            # child chunk, we will keep track of this to build up our
            # full document summary
            if chunk_source not in chunks_by_source:
                chunks_by_source[chunk_source] = []

            chunks_by_source[chunk_source].append(chunk.page_content)

    # Build up the full docs by concatenating all the child chunks
    full_docs_by_id: Dict[str, Document] = {}
    full_doc_ids_by_source: Dict[str, str] = {}
    for source in chunks_by_source:
        chunks_from_source = chunks_by_source[source]
        full_doc_text = "\n".join([c for c in chunks_from_source])
        full_doc_id = hashlib.md5(full_doc_text.encode()).hexdigest()
        full_doc_ids_by_source[source] = full_doc_id
        full_docs_by_id[full_doc_id] = Document(page_content=full_doc_text, metadata={"id": full_doc_id})

    # Associate parent chunks with full docs
    for parent_chunk_id in parent_chunks_by_id:
        parent_chunk = parent_chunks_by_id[parent_chunk_id]
        parent_chunk_source = parent_chunk.metadata.get(SOURCE_KEY)
        if parent_chunk_source:
            full_doc_id = full_doc_ids_by_source.get(parent_chunk_source)
            if full_doc_id:
                parent_chunk.metadata[FULL_DOC_SUMMARY_ID_KEY] = full_doc_id

    full_doc_summaries_by_id = build_full_doc_summary_mappings(full_docs_by_id)
    chunk_summaries_by_id = build_chunk_summary_mappings(parent_chunks_by_id)

    direct_tool_function_name = docset_name_to_direct_retriever_tool_function_name(name)
    direct_tool_description = chunks_to_direct_retriever_tool_description(
        name=name,
        chunks=list(parent_chunks_by_id.values()),
        llm=SMALL_CONTEXT_INSTRUCT_LLM,
        max_chunk_text_length=MAX_CHUNK_TEXT_LENGTH,
    )
    report_details = build_report_details(docset_id)

    if overwrite:
        state = Path(INDEXING_LOCAL_STATE_PATH)
        if state.is_file() and state.exists():
            os.remove(state)

        if get_vector_store_index(docset_id, EMBEDDINGS) is not None:
            del_vector_store_index(docset_id)

    update_local_index(
        docset_id=docset_id,
        full_doc_summaries_by_id=full_doc_summaries_by_id,
        chunks_by_id=parent_chunks_by_id,  # we are using the parent chunks as chunks for expanded context
        direct_tool_function_name=direct_tool_function_name,
        direct_tool_description=direct_tool_description,
        report_details=report_details,
    )

    populate_vector_index(docset_id, chunks=list(chunk_summaries_by_id.values()), overwrite=overwrite)
