from typing import Any
from uuid import uuid4

import os
from langchain_core.documents import Document
from openai import OpenAI
from qdrant_client import QdrantClient, models


# =========================================================
# CONFIG
# =========================================================

EMBEDDING_MODEL = "text-embedding-3-small"

COLLECTION_NAME = "documents"

VECTOR_SIZE = 1536


# =========================================================
# CLIENTS
# =========================================================

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

qdrant_client = QdrantClient(url="http://localhost:6333")


# =========================================================
# CREATE COLLECTION
# =========================================================


def ensure_collection():
    """
    Create Qdrant collection if it does not exist.
    """

    collections = qdrant_client.get_collections()

    exists = any(
        collection.name == COLLECTION_NAME for collection in collections.collections
    )

    if exists:
        return

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )


# =========================================================
# EMBEDDING
# =========================================================


def create_embedding(text: str) -> list[float]:
    """
    Convert text into embedding vector.
    """
    vectors = create_embeddings([text])

    return vectors[0]

def create_embeddings(texts: list[str]):
    # wait_for_rate_limit()
    if not texts:
        return []
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=VECTOR_SIZE,
    )

    return [
        embedding.embedding
        for embedding in response.data
    ]

# =========================================================
# BUILD EMBEDDING TEXT
# =========================================================


def build_embedding_text(
    document_name: str,
    parent_context: list[str],
    node: dict[str, Any],
) -> str:

    parts = []

    # Document name
    if document_name:
        parts.append(f"Document: {document_name}")

    # Parent hierarchy
    for context in parent_context:
        parts.append(context)

    # Current content
    if node["type"] == "paragraph":

        parts.append(f"Content: {node['text']}")

    elif node["type"] == "heading":

        parts.append(f"Heading: {node['text']}")

    elif node["type"] == "title":

        parts.append(f"Title: {node['text']}")

    elif node["type"] == "table":

        headers = node.get("headers", [])
        rows = node.get("rows", [])

        parts.append(f"Headers: {headers}")

        parts.append(f"Rows: {rows}")

    return "\n".join(parts)

def batch_items(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
# =========================================================
# SAVE DOCUMENT TO QDRANT
# =========================================================


def index_document(
    document_id: str,
    document_name: str,
    documents: list[Document],
):

    ensure_collection()

    if not documents:
        return {
            "document_id": document_id,
            "indexed": 0,
        }

    # ---------------------------------------------------------
    # 2. Create embeddings
    # ---------------------------------------------------------

    points = []

    BATCH_SIZE = 20

    for document_list in batch_items(documents, BATCH_SIZE):

        texts = [
            document_item.page_content
            for document_item in document_list
        ]

        vectors = create_embeddings(texts)

        for document_item, vector in zip(document_list, vectors):

            payload = {
                "document_id": document_id,
                "document_name": document_name,
                "metadata": document_item.metadata,
                "text": document_item.page_content,
            }

            point = models.PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload=payload,
            )

            points.append(point)

    # ---------------------------------------------------------
    # 3. Upload points
    # ---------------------------------------------------------

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    return {
        "document_id": document_id,
        "indexed": len(points),
    }