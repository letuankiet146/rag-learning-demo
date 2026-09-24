# Document Parser MVP

A small FastAPI + HTML UI for uploading PDF, DOCX and TXT files.

Pipeline:

Upload
-> parse while preserving structure
-> normalize to a common tree
-> save normalized JSON + original file path/link in SQLite
-> show parsed structure in UI

## Run

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000

## API

- `POST /api/documents` - upload and parse a document
- `GET /api/documents` - list uploaded documents
- `GET /api/documents/{document_id}` - get normalized document
- `GET /api/documents/{document_id}/file` - read the original uploaded file
- `GET /health` - health check

SQLite database: `documents.db`

Uploaded files: `uploads/`

## Normalized shape

The API produces a tree like:

```json
{
  "type": "document",
  "level": 1,
  "text": "iphone17_warranty.pdf",
  "child": [
    {
      "type": "title",
      "level": 2,
      "text": "Warranty policy",
      "child": []
    },
    {
      "type": "heading",
      "level": 3,
      "text": "Warranty Period",
      "child": [
        {
          "type": "paragraph",
          "level": 4,
          "text": "Products are covered for 12 months...",
          "child": []
        },
        {
          "type": "table",
          "level": 4,
          "headers": ["Product", "Warranty", "Price"],
          "rows": [["iPhone 17", "12 months", "$799"]],
          "child": []
        }
      ]
    }
  ]
}
```

## Important MVP decision

PDFs do not reliably contain semantic labels such as "Heading 1". The PDF parser therefore uses layout heuristics (font size, boldness, position and text length) to infer title/heading/paragraph.

DOCX is easier because Word styles such as `Title`, `Heading 1`, and `Heading 2` are available directly.

TXT has no native structure, so simple heuristics are used.

The parser keeps a common normalized structure so later chunking/embedding can operate on one format.
