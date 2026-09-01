# API compatibility

The FastAPI application exposes the same paths and HTTP methods as the API in
the former `rage` package. Run it with `fastapi run src/rage/api/app.py`.

The following behavior cannot be reproduced by this package yet:

- `application/json` collection files return HTTP 501 because the old API used
  `AudioTranscriptionLoader`, which this package does not provide. Add an audio
  transcription loader and register it in the create pipeline to restore it.
- Collections are flat Qdrant collections. `graph_stats`, hierarchical parent
  merging, and child node IDs are unavailable because this package has no
  hierarchical splitter/retriever. Port those components to restore them.
- `language` is stored as metadata, but query translation is not performed.
  `translate_query` is accepted for wire compatibility. Add a translation
  service and apply it before retrieval to restore the old behavior.
- `enable_reranker` and `enable_llm_response` are accepted, but results are not
  reranked and `llm_response` remains null. Add the old reranker and RAG services
  (or replacements) to restore those features.
- Qdrant does not expose the old Weaviate keyword/vector score breakdown.
  `keyword_score`, `vector_score`, and `relative_hybrid_score` therefore contain
  the single score returned by Qdrant.
- `wv_alpha` selects sparse search at `0.0` and hybrid search otherwise. Qdrant's
  current hybrid retriever does not accept the old continuous alpha weighting.
- Collection metadata is reconstructed from the first stored chunk. Empty
  collections therefore return a collection name with empty language and
  document metadata. Add a dedicated metadata store for exact old behavior.
- Object properties and filters address chunk metadata. The API translates old
  property names to Qdrant's `metadata.<property>` payload layout.
