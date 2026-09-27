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
- `POST /api/chat` - ask question base on RAG   
- `GET /health` - health check

SQLite database: `documents.db`

Uploaded files: `uploads/`

## Normalized shape

The API produces a tree like:

```json
{
  "type": "document",
  "level": 1,
  "text": "1bdb908b-7f1f-4ec9-9351-72b7d50e792c.pdf",
  "child": [
    {
      "type": "title",
      "level": 2,
      "text": "Warranty Disclaimer Template",
      "child": [
        {
          "type": "heading",
          "level": 3,
          "text": "Warranty",
          "child": [
            {
              "type": "paragraph",
              "level": 4,
              "text": "Thank you for your interest in the products and services of My Company (change this) .",
              "child": [],
              "page": 1
            },
            {
              "type": "paragraph",
              "level": 4,
              "text": "This Limited Warranty applies to physical goods, and only for physical goods, purchased from My Company (change this) (the \"Physical Goods\").",
              "child": [],
              "page": 1
            }
          ],
          "page": 1
        },
        {
          "type": "heading",
          "level": 3,
          "text": "What does this limited warranty cover?",
          "child": [
            {
              "type": "paragraph",
              "level": 4,
              "text": "This Limited Warranty covers any defects in material or workmanship under normal use during the Warranty Period.",
              "child": [],
              "page": 1
            },
            {
              "type": "paragraph",
              "level": 4,
              "text": "During the Warranty Period, My Company (change this) will repair or replace, at no charge, products or parts of a product that proves defective because of improper material or workmanship, under normal use and maintenance.",
              "child": [],
              "page": 1
            }
          ],
          "page": 1
        },
        {
          "type": "heading",
          "level": 3,
          "text": "What will we do to correct problems?",
          "child": [
            {
              "type": "paragraph",
              "level": 4,
              "text": "My Company (change this) will either repair the Product at no charge, using new or refurbished replacement parts.",
              "child": [],
              "page": 1
            }
          ],
          "page": 1
        },
        {
          "type": "heading",
          "level": 3,
          "text": "How long does the coverage last?",
          "child": [
            {
              "type": "paragraph",
              "level": 4,
              "text": "The Warranty Period for Physical Goods purchased from My Company (change this) is 180 days (change this) from the date of purchase.",
              "child": [],
              "page": 1
            },
            {
              "type": "paragraph",
              "level": 4,
              "text": "A replacement Physical Good or part assumes the remaining warranty of the original Physical Good or 180 days (change this) from the date of replacement or repair, whichever is longer.",
              "child": [],
              "page": 1
            }
          ],
          "page": 1
        },
        {
          "type": "heading",
          "level": 3,
          "text": "What does this limited warranty not cover?",
          "child": [
            {
              "type": "paragraph",
              "level": 4,
              "text": "This Limited Warranty does not cover any problem that is caused by:",
              "child": [],
              "page": 1
            },
            {
              "type": "paragraph",
              "level": 4,
              "text": "● conditions, malfunctions or damage not resulting from defects in material or workmanship",
              "child": [],
              "page": 1
            }
          ],
          "page": 1
        },
        {
          "type": "heading",
          "level": 3,
          "text": "What do you have to do?",
          "child": [
            {
              "type": "paragraph",
              "level": 4,
              "text": "To obtain warranty service, you must first contact us to determine the problem and the most appropriate solution for you.",
              "child": [],
              "page": 1
            }
          ],
          "page": 1
        }
      ],
      "page": 1
    }
  ]
}
```

## Important MVP decision

PDFs do not reliably contain semantic labels such as "Heading 1". The PDF parser therefore uses layout heuristics (font size, boldness, position and text length) to infer title/heading/paragraph.

DOCX is easier because Word styles such as `Title`, `Heading 1`, and `Heading 2` are available directly.

TXT has no native structure, so simple heuristics are used.

The parser keeps a common normalized structure so later chunking/embedding can operate on one format.
