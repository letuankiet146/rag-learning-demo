# app/semantic_chunking.py

import re
from typing import List

import numpy as np


def split_sentences(text: str) -> List[str]:
    text = text.strip()

    if not text:
        return []

    sentences = re.split(
        r'(?<=[.!?。！？])\s+',
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def cosine_similarity(
    vector_a: List[float],
    vector_b: List[float],
) -> float:

    a = np.array(vector_a)
    b = np.array(vector_b)

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


def calculate_consecutive_similarities(
    embeddings: List[List[float]],
) -> List[float]:

    similarities = []

    for i in range(len(embeddings) - 1):

        similarity = cosine_similarity(
            embeddings[i],
            embeddings[i + 1],
        )

        similarities.append(similarity)

    return similarities


def find_break_points(
    similarities: List[float],
    z_threshold: float = -1.0,
) -> List[int]:

    if len(similarities) < 2:
        return []

    mean = np.mean(similarities)
    std = np.std(similarities)

    if std == 0:
        return []

    break_points = []

    for i, similarity in enumerate(similarities):

        z_score = (
            similarity - mean
        ) / std

        if z_score < z_threshold:
            break_points.append(i)

    return break_points


def build_chunks(
    sentences: List[str],
    break_points: List[int],
    max_sentences: int = 5,
) -> List[str]:

    chunks = []
    current_chunk = []

    break_points = set(break_points)

    for i, sentence in enumerate(sentences):

        current_chunk.append(sentence)

        is_break = i in break_points

        reached_max = (
            len(current_chunk) >= max_sentences
        )

        if is_break or reached_max:

            chunks.append(
                " ".join(current_chunk)
            )

            current_chunk = []

    if current_chunk:
        chunks.append(
            " ".join(current_chunk)
        )

    return chunks