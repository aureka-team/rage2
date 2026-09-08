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
| RAGE | Stores collections, maintains their metadata, and retrieves relevant content. |

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
metadata collections. When `enable_llm_response` is true, RAGE reranks the
retrieved chunks and uses the relevant chunks to generate `llm_response`.
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

## Implementation

- [RAGE API](../src/rage/api/app.py)
- [RAGE collection creation](../src/rage/api/routers/collection/create/create_collection.py)
- [RAGE retrieval](../src/rage/api/routers/retriever/retrieve.py)
