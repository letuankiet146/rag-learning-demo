import json
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path("documents.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            content_type TEXT,
            original_path TEXT NOT NULL,
            original_url TEXT NOT NULL,
            normalized_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        '''
    )
    conn.commit()
    conn.close()

def list_documents():
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, filename, content_type, original_url, created_at "
        "FROM documents ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_document(document_id: str):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM documents WHERE id = ?", (document_id,)
    ).fetchone()
    conn.close()

    if not row:
        return None

    result = dict(row)
    result["normalized"] = json.loads(result.pop("normalized_json"))
    return result
