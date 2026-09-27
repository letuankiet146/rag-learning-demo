import os
from openai import OpenAI
from qdrant_client import QdrantClient

COLLECTION_NAME = "documents"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-mini"

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
qdrant_client = QdrantClient("http://localhost:6333")


def create_query_embedding(question: str):
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
    )
    # response = google_client.models.embed_content(
    #     model=EMBEDDING_MODEL,
    #     contents=question,
    # )

    return  response.data[0].embedding


def search_qdrant(question: str, limit: int = 5):
    query_vector = create_query_embedding(question)

    result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    return result.points


def build_context(points):
    contexts = []

    for point in points:
        payload = point.payload or {}

        contexts.append(f"""
Document: {payload.get("document_name", "")}
Section: {payload.get("section", "")}
Page: {payload.get("page", "")}

Content:
{payload.get("text", "")}
""")

    return "\n---\n".join(contexts)


def generate_answer(question: str, context: str):
    prompt = f"""
You are a customer support assistant.

Answer the user's question using the provided context.

Rules:
- Use the context as the primary source of information.
- Do not invent information that is not supported by the context.
- If the context does not contain enough information, say that you do not have enough information.
- Answer clearly and concisely.

Context:
{context}

User question:
{question}
"""

    response = openai_client.responses.create(
        model=CHAT_MODEL,
        input=prompt,
    )

    return response.output_text


def chat(question: str, limit: int = 5):
    points = search_qdrant(
        question=question,
        limit=limit,
    )


    context = build_context(points)
    print(context)
    answer = generate_answer(
        question=question,
        context=context,
    )

    sources = []

    for point in points:
        payload = point.payload or {}

        sources.append(
            {
                "score": point.score,
                "document_name": payload.get("document_name"),
                "page": payload.get("page"),
                "section": payload.get("section"),
                "text": payload.get("text"),
            }
        )

    return {
        "answer": answer,
        "sources": sources,
    }
