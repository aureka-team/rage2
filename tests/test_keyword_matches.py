from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from fastembed.sparse.bm25 import Bm25
from langchain_qdrant import FastEmbedSparse

from rage.utils.keyword_matches import keyword_match_ranges


class KeywordMatchRangesTest(TestCase):
    def test_stemmed_matches_use_original_utf16_offsets(self):
        with TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "english.txt").write_text("the\n", encoding="utf-8")
            with patch.object(Bm25, "download_model", return_value=model_path):
                embeddings = FastEmbedSparse(cache_dir=directory)

        model = embeddings._model.model
        self.assertIsInstance(model, Bm25)
        ranges = keyword_match_ranges(
            "🎙 Collective response, the collection.",
            model,
            set(embeddings.embed_query("the collection").indices),
        )
        self.assertEqual(ranges, [(3, 13), (28, 38)])
