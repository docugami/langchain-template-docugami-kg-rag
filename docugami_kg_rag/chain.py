import sys
from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.schema.runnable import Runnable, RunnableLambda
from langchain.tools.base import BaseTool
from langchain.tools.render import render_text_description

from docugami_kg_rag.config import LARGE_CONTEXT_LLM, DEFAULT_USE_REPORTS
from docugami_kg_rag.helpers.indexing import read_all_local_index_state
from docugami_kg_rag.helpers.reports import get_retrieval_tool_for_report
from docugami_kg_rag.helpers.retrieval import get_retrieval_tool_for_docset


def _get_tools(use_reports=DEFAULT_USE_REPORTS) -> List[BaseTool]:
    """
    Build retrieval tools.
    """

    local_state = read_all_local_index_state()

    docset_retrieval_tools: List[BaseTool] = []
    report_retrieval_tools: List[BaseTool] = []
    for docset_id in local_state:
        docset_state = local_state[docset_id]
        direct_retrieval_tool = get_retrieval_tool_for_docset(docset_id, docset_state)
        if direct_retrieval_tool:
            # Direct retrieval tool for each indexed docset (direct KG-RAG against semantic XML)
            docset_retrieval_tools.append(direct_retrieval_tool)

        for report in docset_state.reports:
            # Report retrieval tool for each published report (user-curated views on semantic XML)
            report_retrieval_tool = get_retrieval_tool_for_report(report)
            if report_retrieval_tool:
                report_retrieval_tools.append(report_retrieval_tool)

    tools = docset_retrieval_tools

    if use_reports:
        tools = tools + report_retrieval_tools

    return tools


def _llm() -> Runnable:
    return RunnableLambda(lambda x: x["input"]) | LARGE_CONTEXT_LLM.bind(stop=["\nObservation"])


# setup ReAct style prompt
prompt = hub.pull("hwchase17/react-json")
tools = _get_tools()
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    }
    | prompt
    | _llm
    | ReActJsonSingleInputOutputParser()
)


class AgentInput(BaseModel):
    input: str = ""
    use_reports: bool = Field(
        default=DEFAULT_USE_REPORTS,
        extra={"widget": {"type": "bool", "input": "input", "output": "output"}},
    )


chain = AgentExecutor(
    agent=agent,
    tools=_get_tools(),
    verbose=False,
    handle_parsing_errors=True,
).with_types(
    input_type=AgentInput,
)


if __name__ == "__main__":
    if sys.gettrace():
        # This code will only run if a debugger is attached

        output = chain.invoke(
            {
                "input": "What happened in Yelm, Washington?",
            }
        )

        print(output)
