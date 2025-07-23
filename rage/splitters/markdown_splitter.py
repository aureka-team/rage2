from langchain_text_splitters.markdown import MarkdownTextSplitter
from .token_splitter import TokenSplitter


class MarkdownSplitter(TokenSplitter):
    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 25,
    ):
        super().__init__()

        self.splitter = MarkdownTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
