import os
from pathlib import Path
from langchain_core.documents import Document
import tiktoken
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from llama_cloud import LlamaCloud

# 2. Cấu hình giới hạn
TARGET_CHUNK_SIZE = 400  # tokens
CHUNK_OVERLAP = 50       # tokens

# 1. Định nghĩa bộ đếm Token chuẩn cho LLM/Embedding
tokenizer = tiktoken.get_encoding("cl100k_base")

def token_len(text: str) -> int:
    return len(tokenizer.encode(text))

def chunking_md(md_content: str) -> list[Document]:
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    header_docs = markdown_splitter.split_text(md_content)

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TARGET_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=token_len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    final_chunks = []
    for doc in header_docs:
        if token_len(doc.page_content) > TARGET_CHUNK_SIZE:
            sub_chunks = recursive_splitter.split_documents([doc])
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(doc)

    return final_chunks
    
# ---------------- PDF ----------------
def llma_parsing(path: Path) -> list[Document]:
    file_path_str = str(path)
    client = LlamaCloud(api_key=os.environ["LLMAPARSE_API_KEY"])

    # Upload
    file_obj = client.files.create(file=file_path_str, purpose="parse")

    # Submit + poll + get (parsing.parse wraps create / wait_for_completion / get)
    # Raises on FAILED or CANCELLED. Tune polling_interval=, timeout= if needed.
    result = client.parsing.parse(
        file_id=file_obj.id,
        # The parsing tier. Options: fast, cost_effective, agentic, agentic_plus,
        tier="agentic",
        # The version of the parsing tier to use. Use 'latest' for the most recent version,
        version="latest",
        # expand: which fields to materialize (markdown_full, text_full, items, *_content_metadata, ...),
        expand=["markdown_full", "text_full"],
    )
    md_content = result.markdown_full or ""   
    return chunking_md(md_content)