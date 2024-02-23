import os
from pathlib import Path
import pickle
import random
from typing import Dict, List, Optional
import requests

from docugami import Docugami

from langchain_core.documents import Document
from langchain.storage.in_memory import InMemoryStore

from docugami_langchain.document_loaders.docugami import DocugamiLoader
from docugami_langchain.retrievers.mappings import (
    build_full_doc_summary_mappings,
    build_chunk_summary_mappings,
    build_doc_maps_from_chunks,
)
from docugami_langchain.tools.retrieval import (
    summaries_to_direct_retriever_tool_description,
    docset_name_to_direct_retriever_tool_function_name,
)
from docugami_langchain.tools.reports import (
    connect_to_db,
    excel_to_sqlite_connection,
    report_name_to_report_query_tool_function_name,
    report_details_to_report_query_tool_description,
)

from docugami_kg_rag.config import (
    EXAMPLES_PATH,
    INCLUDE_XML_TAGS,
    INDEXING_LOCAL_STATE_PATH,
    MAX_CHUNK_TEXT_LENGTH,
    MIN_CHUNK_TEXT_LENGTH,
    MAX_FULL_DOCUMENT_TEXT_LENGTH,
    PARENT_HIERARCHY_LEVELS,
    SUB_CHUNK_TABLES,
    EMBEDDINGS,
    LARGE_CONTEXT_INSTRUCT_LLM,
    SMALL_CONTEXT_INSTRUCT_LLM,
    get_vector_store_index,
    init_vector_store_index,
    del_vector_store_index,
)
from docugami_kg_rag.state_models import ReportDetails, LocalIndexState
from docugami_kg_rag.config import DOCUGAMI_API_KEY, INDEXING_LOCAL_REPORT_DBS_ROOT


HEADERS = {"Authorization": f"Bearer {DOCUGAMI_API_KEY}"}


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

    full_docs_by_id, parent_chunks_by_id = build_doc_maps_from_chunks(chunks)

    full_doc_summaries_by_id = build_full_doc_summary_mappings(
        docs_by_id=full_docs_by_id,
        llm=LARGE_CONTEXT_INSTRUCT_LLM,
        embeddings=EMBEDDINGS,
        include_xml_tags=INCLUDE_XML_TAGS,
        summarize_document_examples_file=EXAMPLES_PATH / "summarize_document_examples.yaml",
    )
    chunk_summaries_by_id = build_chunk_summary_mappings(
        docs_by_id=parent_chunks_by_id,
        llm=SMALL_CONTEXT_INSTRUCT_LLM,
        embeddings=EMBEDDINGS,
        include_xml_tags=INCLUDE_XML_TAGS,
        summarize_chunk_examples_file=EXAMPLES_PATH / "summarize_chunk_examples.yaml",
    )

    direct_tool_function_name = docset_name_to_direct_retriever_tool_function_name(name)
    direct_tool_description = summaries_to_direct_retriever_tool_description(
        name=name,
        summaries=random.sample(
            list(full_doc_summaries_by_id.values()), min(len(full_doc_summaries_by_id), 3)
        ),  # give 3 randomly selected summaries summaries
        llm=SMALL_CONTEXT_INSTRUCT_LLM,
        embeddings=EMBEDDINGS,
        max_sample_documents_cutoff_length=MAX_FULL_DOCUMENT_TEXT_LENGTH,
        describe_document_set_examples_file=EXAMPLES_PATH / "describe_document_set_examples.yaml",
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


def download_project_latest_xlsx(project_url: str, local_xlsx: Path) -> Optional[Path]:
    response = requests.request(
        "GET",
        project_url + "/artifacts/latest?name=spreadsheet.xlsx",
        headers=HEADERS,
        data={},
    )
    if response.ok:
        response_json = response.json()["artifacts"]
        xlsx_artifact = next(
            (item for item in response_json if str(item["name"]).lower().endswith(".xlsx")),
            None,
        )
        if xlsx_artifact:
            artifact_id = xlsx_artifact["id"]
            response = requests.request(
                "GET",
                project_url + f"/artifacts/latest/{artifact_id}/content",
                headers=HEADERS,
                data={},
            )
            if response.ok:
                os.makedirs(str(local_xlsx.parent), exist_ok=True)
                with open(local_xlsx, "wb") as f:
                    f.write(response.content)
                    return local_xlsx
            else:
                raise Exception(
                    f"Failed to download XLSX for {project_url}",
                )
    elif response.status_code == 404:
        # No artifacts found: this project has never been published
        return None
    else:
        raise Exception(f"Failed to download XLSX for {project_url}")


def build_report_details(docset_id: str) -> List[ReportDetails]:
    docugami_client = Docugami()

    projects_response = docugami_client.projects.list()
    if not projects_response or not projects_response.projects:
        return []  # no projects found

    projects = [p for p in projects_response.projects if p.docset.id == docset_id]
    details: List[ReportDetails] = []
    for project in projects:
        local_xlsx_path = download_project_latest_xlsx(
            project.url, Path(INDEXING_LOCAL_REPORT_DBS_ROOT) / f"{project.id}.xlsx"
        )
        if local_xlsx_path:
            report_name = project.name or local_xlsx_path.name
            conn = excel_to_sqlite_connection(local_xlsx_path, report_name)
            db = connect_to_db(conn)
            table_info = db.get_table_info()
            details.append(
                ReportDetails(
                    id=project.id,
                    name=report_name,
                    local_xlsx_path=local_xlsx_path,
                    retrieval_tool_function_name=report_name_to_report_query_tool_function_name(project.name),
                    retrieval_tool_description=report_details_to_report_query_tool_description(project.name, table_info),
                )
            )

    return details
