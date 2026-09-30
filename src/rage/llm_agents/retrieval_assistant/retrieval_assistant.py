from pathlib import Path

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
    system_prompt=Path(__file__).with_name("system-prompt.md").read_text(),
    output_type=NativeOutput(RetrievalAssistantOutput),
    retries=3,
)
