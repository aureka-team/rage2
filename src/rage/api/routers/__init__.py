from .collection.create.create_collection import collection_create_router
from .collection.get.get_collection import collection_get_router
from .collection.get.get_collection_objects import collection_get_objects_router
from .collection.list.list_collections import collection_list_router
from .collection.remove.remove_collection import collection_remove_router
from .retriever.retrieve import retrieve_router


__all__ = [
    "collection_create_router",
    "collection_get_objects_router",
    "collection_get_router",
    "collection_list_router",
    "collection_remove_router",
    "retrieve_router",
]
