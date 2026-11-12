import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from markitdown import MarkItDown

from rage.loaders.markdown_loader import MarkdownLoader, _get_markitdown


class MarkdownLoaderTest(IsolatedAsyncioTestCase):
    async def test_concurrent_loaders_reuse_converter_and_preserve_content(self):
        _get_markitdown.cache_clear()
        self.addCleanup(_get_markitdown.cache_clear)

        with TemporaryDirectory() as directory:
            paths = [Path(directory) / f"document-{index}.txt" for index in range(20)]
            for index, path in enumerate(paths):
                path.write_text(f"Document {index}", encoding="utf-8")

            with patch(
                "rage.loaders.markdown_loader.MarkItDown",
                wraps=MarkItDown,
            ) as converter:
                results = await asyncio.gather(
                    *(MarkdownLoader().load(source_path=str(path)) for path in paths)
                )

            converter.assert_called_once_with()

        self.assertEqual(
            [documents[0].text for documents in results],
            [f"Document {index}" for index in range(20)],
        )

