from pathlib import Path

from llm_agents.meta.interfaces import LLMAgent
from pydantic import BaseModel, Field, StrictStr
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.openai import OpenAIChatModelSettings


class RetrievalAssistantOutput(BaseModel):
    response: StrictStr | None = Field(
        default=None,
        description="The answer based on the relevant text chunks.",
    )


agent = Agent(
    name="retrieval-assistant",
    model="openai-chat:gpt-5.6-terra",
    model_settings=OpenAIChatModelSettings(openai_reasoning_effort="none"),
    system_prompt=LLMAgent.read_file(
        file_path=str(Path(__file__).with_name("system-prompt.md"))
    ),
    output_type=NativeOutput(RetrievalAssistantOutput),
    retries=3,
)


class RetrievalAssistant(LLMAgent[None, RetrievalAssistantOutput]):
    def __init__(self, max_concurrency: int = 10):
        super().__init__(agent=agent, max_concurrency=max_concurrency)
