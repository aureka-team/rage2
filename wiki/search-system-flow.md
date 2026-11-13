---
type: Architecture
title: Search system flow
description: Request flows from Lumos to RAGE, directly and through Diagon.
tags:
  - diagon
  - lumos
  - rage
  - search
---

# Search system flow

Search functionality spans Lumos, Diagon, and RAGE. Lumos owns the
application data and initiates indexing and search operations. Diagon
orchestrates transcription indexing. RAGE stores indexed content and performs
retrieval.

## Service responsibilities

| Service | Responsibility |
| --- | --- |
| Lumos | Initiates transcription indexing, synchronizes annotations and metadata, and sends search queries. |
| Diagon | Runs the transcription-indexing workflow and coordinates creation of the corresponding RAGE collection. |
| RAGE | Stores chunks in one Qdrant collection, maintains a resource registry, and retrieves relevant content. |

## Lumos to RAGE

Lumos communicates directly with RAGE for search and for annotation
and metadata synchronization.

```text
Search:
Lumos -> RAGE POST /rage/retriever/retrieve

Annotation and metadata synchronization:
Lumos -> RAGE POST /rage/collection/create
      -> RAGE POST /rage/collection/remove
```

`POST /rage/retriever/retrieve` searches the transcription, annotation, and
metadata resources in a single Qdrant query. Lumos sends the IDs it has already
authorized. RAGE resolves those IDs in one metadata lookup and filters the
query to the current generation of each resource. The API uses the existing
`collection_name` and `collection_names` fields for those resource IDs. When `enable_llm_response` is true, RAGE reranks the
retrieved chunks and uses the relevant chunks to generate `llm_response`.
The response lists every missing collection in `invalid_collections`, preserving
each identifier and its request order. `error` is true whenever that list is
nonempty. RAGE still returns search results from existing collections in a mixed
request. When `error` is false, an empty `retriever_items` list means the
requested collections exist but the search found no matches.

For sparse BM25 search, each result includes `match_ranges` with half-open
UTF-16 offsets into its original `text`. RAGE uses the active BM25 model's
token IDs, so a query for `collection` can highlight `collective`.
The field is null when match ranges are unavailable.

Lumos creates, replaces, or removes annotation and metadata collections as its
application data changes.

## Lumos to Diagon to RAGE

Lumos delegates transcription indexing to Diagon. Diagon converts the Aureka
transcription into the JSON document sent to RAGE, computes its checksum, and
coordinates collection creation.

```text
Lumos -> Diagon -> RAGE POST /rage/collection/get
                 -> RAGE POST /rage/collection/create (only when content changed)
                 -> RAGE POST /rage/collection/get
       <- Diagon <- RAGE collection metadata
```

The indexing decision follows the current RAGE collection state:

1. When the collection exists and its stored file checksum matches, Diagon
   returns that response without creating the collection again.
2. When the collection is missing, Diagon calls `/rage/collection/create`.
3. When the collection exists but its checksum differs, Diagon calls
   `/rage/collection/create` with `overwrite=true`.
4. After creation, Diagon calls `/rage/collection/get` and returns the collection
   metadata to Lumos. Lumos records the checksum and marks search
   synchronization as complete.

## Active RAGE endpoints

| Endpoint | Caller | Purpose |
| --- | --- | --- |
| `POST /rage/collection/get` | Diagon | Checks a transcription collection before indexing and retrieves its metadata afterward. |
| `POST /rage/collection/create` | Diagon and Lumos | Diagon indexes transcriptions; Lumos directly indexes annotations and metadata. |
| `POST /rage/collection/remove` | Lumos | Deletes or replaces annotation and metadata collections. |
| `POST /rage/retriever/retrieve` | Lumos | Performs search across transcription, annotation, and metadata collections. |

`POST /rage/collection/list` exists, but currently has no active application call
site in these flows.

## Rebuild the shared index

Deploy the new RAGE API before rebuilding the index. The new API uses
`rage_content_v1` for chunks and a separate registry whose name is
`COLLECTION_METADATA` with the suffix `_shared_v1`. Legacy indexes and
checksums are outside this lookup path.

Run the following from Lumos against the target environment. Use a fresh
checkpoint for this rebuild, then reuse it to resume the same run.

```bash
uv run lumos-admin reindex-rage --all --checkpoint ./rage-shared-reindex.sqlite
```

Search becomes available for each resource as its indexing completes.
Verify searches across transcripts, annotations, and metadata, including
restricted resources, before reclaiming the old Qdrant storage. Remove old
collections through Qdrant administration after the cutover. Keep
`rage_content_v1` and the shared registry.

Run the RAGE HTTP API with one worker and one replica. Resource writes and
deletes use in-process locks. A replacement becomes searchable when the
registry points to its new generation, after all its chunks have been stored.

## Implementation

- [RAGE API](../src/rage/api/app.py)
- [RAGE collection creation](../src/rage/api/routers/collection/create/create_collection.py)
- [RAGE retrieval](../src/rage/api/routers/retriever/retrieve.py)
