from pathlib import Path

from llm_agents.meta.interfaces import LLMAgent
from pydantic import BaseModel, Field, PositiveInt, StrictStr
from pydantic_ai import Agent, ToolOutput
from pydantic_ai.models.openai import OpenAIChatModelSettings


class TextChunk(BaseModel):
    chunk_id: PositiveInt
    text: StrictStr


class RerankerOutput(BaseModel):
    relevant_chunk_ids: list[PositiveInt] = Field(
        default_factory=list,
        description=(
            "Relevant chunk IDs ordered from most to least relevant to the query."
        ),
    )


agent = Agent(
    name="reranker",
    model="openai-chat:gpt-5.6-luna",
    model_settings=OpenAIChatModelSettings(openai_reasoning_effort="none"),
    system_prompt=LLMAgent.read_file(
        file_path=str(Path(__file__).with_name("system-prompt.md"))
    ),
    output_type=ToolOutput(RerankerOutput),
    retries=3,
)


class Reranker(LLMAgent[None, RerankerOutput]):
    def __init__(self, max_concurrency: int = 10):
        super().__init__(agent=agent, max_concurrency=max_concurrency)
