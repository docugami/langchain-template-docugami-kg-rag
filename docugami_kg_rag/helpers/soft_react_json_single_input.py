from typing import Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser


class SoftReActJsonSingleInputOutputParser(ReActJsonSingleInputOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:

        try:
            return super().parse(text)
        except OutputParserException as exc:
            if "Could not parse LLM output:" not in str(exc):
                raise exc

            # just finish on unparseable output and send that to the user
            return AgentFinish({"output": text}, text)

    @property
    def _type(self) -> str:
        return "soft-react-json-single-input"
