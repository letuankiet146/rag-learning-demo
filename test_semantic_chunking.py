from app.semantic_chunking import (
    split_sentences,
    build_chunks,
)

from app.embedding import (
    embed_sentences,
    calculate_consecutive_similarities,
    find_break_points,
)


text = """
Thời kỳ trước khi bắt nguồn diễn ra trước năm 1750.
Hoạt động bán hàng đã xuất hiện từ thời Phoenicy.
Những người bán dạo thường đi từ nhà này sang nhà khác.
Đến thời La Mã, các nhà buôn trở nên phổ biến.
Cách mạng công nghiệp bắt đầu vào khoảng năm 1750.
Năng lực sản xuất tăng mạnh.
Các doanh nghiệp bắt đầu phát triển hoạt động bán hàng.
"""


sentences = split_sentences(text)

print("\nSENTENCES:")
for i, sentence in enumerate(sentences):
    print(i, sentence)


embeddings = embed_sentences(sentences)

similarities = calculate_consecutive_similarities(
    embeddings
)

print("\nSIMILARITIES:")

for i, similarity in enumerate(similarities):
    print(
        f"S{i} ↔ S{i+1}: "
        f"{similarity:.4f}"
    )


break_points = find_break_points(
    similarities,
    z_threshold=-1.0,
)

print("\nBREAK POINTS:")
print(break_points)


chunks = build_chunks(
    sentences,
    break_points,
)

print("\nFINAL CHUNKS:")

for i, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {i+1} ---")
    print(chunk)    