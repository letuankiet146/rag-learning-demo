from app.parsers import (
    chunking_md,
    token_len
)
from pathlib import Path

md_path = Path("./example/llmaparse.md")
content = md_path.read_text(encoding='utf-8')
elements = chunking_md(content)