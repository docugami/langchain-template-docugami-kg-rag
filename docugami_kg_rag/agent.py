import sys
from typing import List, Optional, Union

from docugami_langchain.agents import ReActAgent
from docugami_langchain.tools.common import get_generic_tools
from docugami_langchain.tools.reports import get_retrieval_tool_for_report
from docugami_langchain.tools.retrieval import get_retrieval_tool_for_docset
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from docugami_kg_rag.config import (
    DEFAULT_USE_CONVERSATIONAL_TOOLS,
    DEFAULT_USE_REPORTS,
    EMBEDDINGS,
    EXAMPLES_PATH,
    LARGE_CONTEXT_INSTRUCT_LLM,
    RERANKER,
    RETRIEVER_K,
    SMALL_CONTEXT_INSTRUCT_LLM,
    SQL_GEN_LLM,
    get_vector_store_index,
)
from docugami_kg_rag.indexing import read_all_local_index_state


def build_tools(
    use_reports: bool = DEFAULT_USE_REPORTS,
    use_conversation_tools: bool = DEFAULT_USE_CONVERSATIONAL_TOOLS,
) -> List[BaseTool]:
    """
    Build retrieval tools.
    """

    local_state = read_all_local_index_state()

    tools: List[BaseTool] = []
    for docset_id in local_state:
        docset_state = local_state[docset_id]
        chunk_vectorstore = get_vector_store_index(docset_id, EMBEDDINGS)

        if chunk_vectorstore is not None:

            def _fetch_parent_doc_callback(key: str) -> Optional[str]:
                results = docset_state.chunks_by_id.mget([key])
                if results and results[0]:
                    first_result: Document = results[0]
                    return first_result.page_content
                return None

            def _fetch_full_doc_summary_callback(key: str) -> Optional[str]:
                results = docset_state.full_doc_summaries_by_id.mget([key])
                if results and results[0]:
                    first_result: Document = results[0]
                    return first_result.page_content
                return None

            direct_retrieval_tool = get_retrieval_tool_for_docset(
                chunk_vectorstore=chunk_vectorstore,
                retrieval_tool_function_name=docset_state.retrieval_tool_function_name,
                retrieval_tool_description=docset_state.retrieval_tool_description,
                llm=LARGE_CONTEXT_INSTRUCT_LLM,
                embeddings=EMBEDDINGS,
                re_ranker=RERANKER,
                fetch_parent_doc_callback=_fetch_parent_doc_callback,
                fetch_full_doc_summary_callback=_fetch_full_doc_summary_callback,
                retrieval_k=RETRIEVER_K,
            )
            if direct_retrieval_tool:
                # Direct retrieval tool for each indexed docset (direct KG-RAG against semantic XML)
                tools.append(direct_retrieval_tool)

        if use_reports:
            for report in docset_state.reports:
                # Report retrieval tool for each published report (user-curated views on semantic XML)
                report_retrieval_tool = get_retrieval_tool_for_report(
                    local_xlsx_path=report.local_xlsx_path,
                    report_name=report.name,
                    retrieval_tool_function_name=report.retrieval_tool_function_name,
                    retrieval_tool_description=report.retrieval_tool_description,
                    sql_llm=SQL_GEN_LLM,
                    embeddings=EMBEDDINGS,
                    sql_fixup_examples_file=EXAMPLES_PATH / "sql_fixup_examples.yaml",
                    sql_examples_file=EXAMPLES_PATH / "sql_examples.yaml",
                )
                if report_retrieval_tool:
                    tools.append(report_retrieval_tool)

        if use_conversation_tools:
            tools += get_generic_tools(
                llm=SMALL_CONTEXT_INSTRUCT_LLM,
                embeddings=EMBEDDINGS,
                answer_examples_file=EXAMPLES_PATH / "answer_examples.yaml",
            )

    return tools


class AgentInput(BaseModel):
    messages: List[Union[HumanMessage, AIMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )


agent = (
    ReActAgent(llm=LARGE_CONTEXT_INSTRUCT_LLM, embeddings=EMBEDDINGS, tools=build_tools())
    .runnable()
    .with_types(
        input_type=AgentInput,  # type: ignore
    )
)
if __name__ == "__main__":
    if sys.gettrace():
        # This code will only run if a debugger is attached

        output = agent.invoke(
            {
                "question": "What is the project number for the contract with snelson?",
                "chat_history": [],
            }
        )

        print(output)
