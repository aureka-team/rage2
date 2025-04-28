from pydantic import BaseModel, StrictStr, Field

from common.cache import RedisCache

from rage.conf import llm_agents
from llm_agents.meta.interfaces import LLMAgent


class TableExtractorInput(BaseModel):
    html_table_text: StrictStr


class TableExtractorOutput(BaseModel):
    json_table: StrictStr = Field(
        "A hierarchical and structured JSON-formatted dictionary representation of the parsed HTML table."
    )


class TableExtractor(LLMAgent[TableExtractorInput, TableExtractorOutput]):
    def __init__(
        self,
        conf_path=f"{llm_agents.__path__[0]}/table-extractor.yml",
        max_concurrency: int = 10,
        cache: RedisCache = None,
    ):
        super().__init__(
            conf_path=conf_path,
            agent_input=TableExtractorInput,
            agent_output=TableExtractorOutput,
            max_concurrency=max_concurrency,
            cache=cache,
        )
