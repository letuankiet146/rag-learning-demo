from typing import Any
from uuid import uuid4

import os
from google import genai
from google.genai import types
from qdrant_client import QdrantClient, models
from .semantic_chunking import (
    split_sentences,
    calculate_consecutive_similarities,
    find_break_points,
    build_chunks,
)

# =========================================================
# CONFIG
# =========================================================

# EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MODEL = "gemini-embedding-001"

COLLECTION_NAME = "documents"

VECTOR_SIZE = 3072


# =========================================================
# CLIENTS
# =========================================================

google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

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

    response = google_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    return response.embeddings[0].values


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


# =========================================================
# FLATTEN NORMALIZED TREE
# =========================================================


def flatten_document(
    node: dict[str, Any],
    document_name: str,
    parent_context: list[str] | None = None,
):
    """
    Convert normalized tree into a flat list
    of embedding candidates.
    """

    if parent_context is None:
        parent_context = []

    current_context = list(parent_context)

    node_type = node.get("type")
    text = node.get("text", "")

    # ---------------------------------------------------------
    # Add current title / heading to context
    # ---------------------------------------------------------

    if node_type == "title":
        current_context.append(f"Title: {text}")

    elif node_type == "heading":
        current_context.append(f"Section: {text}")

    # ---------------------------------------------------------
    # Create embedding candidates
    # ---------------------------------------------------------

    results = []

    # ---------------------------------------------------------
    # PARAGRAPH
    # ---------------------------------------------------------

    if node_type == "paragraph":

        # Semantic chunking
        semantic_chunks = semantic_chunk_paragraph(
            text,
            z_threshold=-1.0,
        )

        for chunk_text in semantic_chunks:

            # Create embedding text for THIS final chunk
            embedding_text = build_embedding_text(
                document_name=document_name,
                parent_context=current_context,
                node={
                    **node,
                    "text": chunk_text,
                },
            )

            results.append(
                {
                    "type": "paragraph",
                    "text": chunk_text,
                    "embedding_text": embedding_text,
                    "level": node.get("level"),
                    "page": node.get("page"),
                    "parent_context": current_context,
                }
            )

    # ---------------------------------------------------------
    # TABLE
    # ---------------------------------------------------------

    elif node_type == "table":

        # Table stays as one chunk
        embedding_text = build_embedding_text(
            document_name=document_name,
            parent_context=current_context,
            node=node,
        )

        results.append(
            {
                "type": "table",
                "text": text,
                "embedding_text": embedding_text,
                "level": node.get("level"),
                "page": node.get("page"),
                "parent_context": current_context,
            }
        )

    # ---------------------------------------------------------
    # Process children
    # ---------------------------------------------------------

    for child in node.get("child", []):
        results.extend(
            flatten_document(
                node=child,
                document_name=document_name,
                parent_context=current_context,
            )
        )

    return results


# =========================================================
# SAVE DOCUMENT TO QDRANT
# =========================================================


def index_document(
    document_id: str,
    document_name: str,
    normalized_document: dict[str, Any],
):

    ensure_collection()

    # ---------------------------------------------------------
    # 1. Flatten tree
    # ---------------------------------------------------------

    chunks = flatten_document(
        node=normalized_document,
        document_name=document_name,
    )

    if not chunks:
        return {
            "document_id": document_id,
            "indexed": 0,
        }

    # ---------------------------------------------------------
    # 2. Create embeddings
    # ---------------------------------------------------------

    points = []

    for chunk in chunks:

        vector = create_embedding(chunk["embedding_text"])

        # -----------------------------------------------------
        # Payload
        # -----------------------------------------------------

        payload = {
            "document_id": document_id,
            "document_name": document_name,
            "type": chunk["type"],
            "text": chunk["text"],
            "embedding_text": chunk["embedding_text"],
            "level": chunk["level"],
            "page": chunk["page"],
            "parent_context": chunk["parent_context"],
        }

        # -----------------------------------------------------
        # Qdrant point
        # -----------------------------------------------------

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


def embed_sentences(sentences):

    if not sentences:
        return []

    response = google_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=sentences,
        config= types.EmbedContentConfig(
            task_type="SEMANTIC_SIMILARITY",
            output_dimensionality=VECTOR_SIZE,
        ),
    )

    return [embedding.values for embedding in response.embeddings]


def embed_chunk(
    chunk_text: str,
    title: str,
    heading: str,
):

    embedding_text = f"""
Title: {title}

Section: {heading}

Content:
{chunk_text}
""".strip()

    response = google_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=embedding_text,
        config=types.EmbedContentConfig(
            output_dimensionality=VECTOR_SIZE,
        ),
    )

    return (
        embedding_text,
        response.embeddings[0].values,
    )


def semantic_chunk_paragraph(
    paragraph: str,
    z_threshold: float = -1.0,
):

    sentences = split_sentences(paragraph)

    if len(sentences) <= 1:
        return [paragraph.strip()]

    sentence_embeddings = embed_sentences(sentences)

    similarities = calculate_consecutive_similarities(sentence_embeddings)

    break_points = find_break_points(
        similarities,
        z_threshold=z_threshold,
    )

    chunks = build_chunks(
        sentences,
        break_points,
    )

    return chunks
