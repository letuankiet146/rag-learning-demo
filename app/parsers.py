from pathlib import Path
from typing import Any
import re

import fitz
from docx import Document as DocxDocument


def node(node_type: str, level: int, text: str = "", child=None, **extra):
    result = {
        "type": node_type,
        "level": level,
        "text": text,
        "child": child or [],
    }
    result.update(extra)
    return result


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_document(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix == ".docx":
        return parse_docx(path)
    if suffix in {".txt", ".text"}:
        return parse_txt(path)

    raise ValueError("Unsupported file type. Use PDF, DOCX or TXT.")


# ---------------- DOCX ----------------

def parse_docx(path: Path) -> dict[str, Any]:
    doc = DocxDocument(path)
    root = node("document", 1, path.name)
    title_node = None
    current_heading = None

    for block in iter_docx_blocks(doc):
        if block["type"] == "table":
            table_node = node(
                "table",
                4,
                headers=block["headers"],
                rows=block["rows"],
            )
            attach_node(root, current_heading, table_node)
            continue

        text = clean_text(block["text"])
        if not text:
            continue

        style = block["style"].lower()

        if style == "title":
            title_node = node("title", 2, text)
            root["child"].append(title_node)
            current_heading = None
        elif style.startswith("heading 1"):
            current_heading = node("heading", 3, text)
            root["child"].append(current_heading)
        elif style.startswith("heading 2"):
            current_heading = node("heading", 3, text)
            root["child"].append(current_heading)
        elif style.startswith("heading 3"):
            current_heading = node("heading", 3, text)
            root["child"].append(current_heading)
        else:
            paragraph_node = node("paragraph", 4, text)
            attach_node(root, current_heading, paragraph_node)

    return root


def iter_docx_blocks(doc):
    # Preserve paragraph/table order by walking the document XML body.
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    from docx.document import Document as _Document
    from docx.oxml.text.paragraph import CT_P
    from docx.oxml.table import CT_Tbl

    parent = doc.element.body

    for child in parent.iterchildren():
        if isinstance(child, CT_P):
            paragraph = Paragraph(child, parent)
            yield {
                "type": "paragraph",
                "text": paragraph.text,
                "style": paragraph.style.name if paragraph.style else "",
            }
        elif isinstance(child, CT_Tbl):
            table = Table(child, parent)
            rows = []
            for row in table.rows:
                rows.append([clean_text(cell.text) for cell in row.cells])

            headers = rows[0] if rows else []
            data_rows = rows[1:] if len(rows) > 1 else []
            yield {
                "type": "table",
                "headers": headers,
                "rows": data_rows,
            }


# ---------------- TXT ----------------

def parse_txt(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    root = node("document", 1, path.name)

    lines = [line.rstrip() for line in text.splitlines()]
    non_empty = [line.strip() for line in lines if line.strip()]

    if not non_empty:
        return root

    # TXT has no real semantic structure. Use conservative heuristics.
    first = non_empty[0]
    if len(first) <= 120 and not first.endswith((".", "?", "!", ":")):
        root["child"].append(node("title", 2, first))
        lines = lines[lines.index(next(x for x in lines if x.strip())) + 1:]

    current_heading = None
    paragraph_buffer = []

    def flush():
        nonlocal paragraph_buffer
        text_value = clean_text(" ".join(paragraph_buffer))
        if text_value:
            attach_node(root, current_heading, node(
                "paragraph", 4, text_value))
        paragraph_buffer = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            flush()
            continue

        if is_txt_heading(stripped):
            flush()
            current_heading = node("heading", 3, stripped)
            root["child"].append(current_heading)
        else:
            paragraph_buffer.append(stripped)

    flush()
    return root


def is_txt_heading(text: str) -> bool:
    if len(text) > 100:
        return False
    if re.match(r"^(#{1,6})\s+.+", text):
        return True
    if re.match(r"^\d+(\.\d+)*\s+\S+", text):
        return True
    words = text.split()
    return 1 <= len(words) <= 10 and text.isupper()


# ---------------- PDF ----------------
def parse_pdf(path: Path) -> dict[str, Any]:
    pdf = fitz.open(path)

    root = node(
        "document",
        1,
        path.name
    )

    # =========================================================
    # STEP 1
    # Extract tất cả text blocks từ PDF
    # =========================================================

    all_blocks = []

    for page_number, page in enumerate(pdf, start=1):

        data = page.get_text("dict")

        for block in data.get("blocks", []):

            # Chỉ xử lý text block
            if "lines" not in block:
                continue

            spans = []

            for line in block["lines"]:

                for span in line["spans"]:

                    text = span.get("text", "").strip()

                    if text:
                        spans.append(span)

            if not spans:
                continue

            # -------------------------------------------------
            # Ghép text trong block
            # -------------------------------------------------

            text = clean_text(
                " ".join(
                    span["text"]
                    for span in spans
                )
            )

            if not text:
                continue

            # -------------------------------------------------
            # Font size lớn nhất trong block
            # -------------------------------------------------

            font_size = max(
                float(span.get("size", 0))
                for span in spans
            )

            # -------------------------------------------------
            # Kiểm tra bold
            # -------------------------------------------------

            is_bold = any(
                "bold"
                in str(
                    span.get("font", "")
                ).lower()
                for span in spans
            )
            is_italic = any(
                "italic"
                in str(
                    span.get("font", "")
                ).lower()
                for span in spans
            )
            if is_bold and is_italic:
                style = "bold_italic"
            elif is_bold:
                style = "bold"
            elif is_italic:
                style = "italic"
            else:
                style = "normal"

            all_blocks.append({
                "page": page_number,
                "text": text,
                "font_size": font_size,
                "style": style,
                "bbox": block.get("bbox"),
            })

    # =========================================================
    # STEP 2
    # Không có text
    # =========================================================

    if not all_blocks:
        return root

    # =========================================================
    # STEP 3
    # Build font hierarchy
    #
    # Ví dụ:
    #
    # 10 -> paragraph
    # 14 -> heading
    # 18 -> heading
    # 24 -> title
    # =========================================================

    font_hierarchy = build_font_hierarchy(
        all_blocks
    )

    # =========================================================
    # STEP 4
    # Convert blocks thành các node
    #
    # Lúc này chưa build tree.
    # Chỉ xác định:
    #
    # type
    # level
    # text
    # page
    # =========================================================

    nodes = []

    for block in all_blocks:

        key = (
            round(block["font_size"], 1),
            block["style"]
        )

        rule = font_hierarchy.get(key)

        if rule is None:
            continue

        node_type = rule["type"]

        # -----------------------------------------------------
        # TITLE
        # -----------------------------------------------------

        if node_type == "title":

            new_node = node(
                "title",
                2,
                block["text"],
                page=block["page"]
            )

            nodes.append(new_node)

            continue

        # -----------------------------------------------------
        # HEADING
        # -----------------------------------------------------

        if node_type == "heading":

            new_node = node(
                "heading",
                rule["level"],
                block["text"],
                page=block["page"]
            )

            nodes.append(new_node)

            continue

        # -----------------------------------------------------
        # PARAGRAPH
        # -----------------------------------------------------

        if node_type == "paragraph":

            # Paragraph level sẽ được xác định
            # khi biết parent heading.
            new_node = node(
                "paragraph",
                None,
                block["text"],
                page=block["page"]
            )

            nodes.append(new_node)

            continue

    # =========================================================
    # STEP 5
    # Build tree dựa trên level
    # =========================================================

    stack = [root]

    for current_node in nodes:

        node_type = current_node["type"]

        # -----------------------------------------------------
        # PARAGRAPH
        # -----------------------------------------------------
        #
        # Paragraph nằm dưới heading gần nhất.
        #
        # Nếu không có heading:
        #
        # document
        #   └── paragraph
        #
        # => level 2
        # -----------------------------------------------------

        if node_type == "paragraph":

            parent = stack[-1]

            current_node["level"] = (
                parent["level"] + 1
            )

            parent["child"].append(
                current_node
            )

            continue

        # -----------------------------------------------------
        # TITLE / HEADING
        # -----------------------------------------------------

        current_level = current_node["level"]

        # -----------------------------------------------------
        # Tìm parent phù hợp
        #
        # Ví dụ:
        #
        # stack:
        #
        # document level 1
        # title    level 2
        # heading  level 3
        #
        # gặp heading level 4
        #
        # => parent = heading level 3
        # -----------------------------------------------------

        while (
            stack
            and
            stack[-1]["level"] >= current_level
        ):
            stack.pop()

        # -----------------------------------------------------
        # Nếu stack rỗng thì quay về document
        # -----------------------------------------------------

        if not stack:
            stack = [root]

        parent = stack[-1]

        # -----------------------------------------------------
        # Add node vào parent
        # -----------------------------------------------------

        parent["child"].append(
            current_node
        )

        # -----------------------------------------------------
        # Node hiện tại trở thành parent
        # cho các node tiếp theo
        # -----------------------------------------------------

        stack.append(
            current_node
        )

    return root


def build_font_hierarchy(all_blocks):
    """
    Build hierarchy using:

        font_size + style

    Example:

        (10, "normal") -> paragraph
        (10, "bold")   -> heading
        (14, "normal") -> heading
        (18, "bold")   -> title
    """

    if not all_blocks:
        return {}


    # =========================================================
    # 1. Count (font_size, style)
    # =========================================================

    style_counts = {}

    for block in all_blocks:

        size = round(
            block["font_size"],
            1
        )

        style = block["style"]

        key = (size, style)

        style_counts[key] = (
            style_counts.get(key, 0) + 1
        )


    # =========================================================
    # 2. Find paragraph style
    #
    # Most frequent combination
    # is our paragraph candidate.
    # =========================================================

    paragraph_key = max(
        style_counts,
        key=style_counts.get
    )


    paragraph_size, paragraph_style = (
        paragraph_key
    )


    hierarchy = {}


    hierarchy[paragraph_key] = {
        "type": "paragraph"
    }


    # =========================================================
    # 3. Find candidate heading/title styles
    # =========================================================

    candidates = []

    for key in style_counts:

        size, style = key

        # Font size larger than paragraph
        if size > paragraph_size:

            candidates.append(key)
            continue


        # Same font size but styled
        if size == paragraph_size:

            if style in {
                "bold",
                "italic",
                "bold_italic"
            }:
                candidates.append(key)


    if not candidates:
        return hierarchy


    # =========================================================
    # 4. Largest font size = title
    # =========================================================

    largest_size = max(
        size
        for size, style in candidates
    )


    title_candidates = [
        key
        for key in candidates
        if key[0] == largest_size
    ]


    # Nếu cùng largest size có nhiều style,
    # ưu tiên bold.
    title_key = sorted(
        title_candidates,
        key=lambda key: (
            key[1] != "bold",
            key[1] != "bold_italic"
        )
    )[0]


    hierarchy[title_key] = {
        "type": "title"
    }


    # =========================================================
    # 5. Remaining candidates = headings
    # =========================================================

    heading_candidates = [
        key
        for key in candidates
        if key != title_key
    ]


    # Larger font size = higher heading
    heading_candidates.sort(
        key=lambda key: key[0],
        reverse=True
    )


    for index, key in enumerate(
        heading_candidates,
        start=3
    ):

        hierarchy[key] = {
            "type": "heading",
            "level": index
        }


    return hierarchy

def is_pdf_title(text: str, size: float, base_size: float) -> bool:
    return len(text) <= 140 and size >= base_size * 1.8


def is_pdf_heading(text: str, size: float, base_size: float, bold: bool) -> bool:
    if len(text) > 160:
        return False

    numbered = bool(re.match(r"^\d+(\.\d+)*[.)]?\s+\S+", text))
    short_bold = bold and len(text.split()) <= 15
    larger = size >= base_size * 1.25

    return numbered or short_bold or larger


def attach_node(root, current_heading, child):
    if current_heading is not None:
        current_heading["child"].append(child)
    else:
        root["child"].append(child)
