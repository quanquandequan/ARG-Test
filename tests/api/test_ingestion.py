"""API tests for /documents/ingest and DELETE /documents/{id}."""


def test_ingest_markdown_inserts_chunks(wired_singletons, client):
    body = b"# Title\n\nThis is the body of the document. " * 5
    r = client.post(
        "/documents/ingest",
        files={"file": ("demo.md", body, "text/markdown")},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["chunk_count"] >= 1
    assert wired_singletons["vectordb"].count() == data["chunk_count"]


def test_ingest_empty_upload_rejected(client):
    r = client.post(
        "/documents/ingest",
        files={"file": ("empty.md", b"", "text/markdown")},
    )
    assert r.status_code == 400


def test_ingest_unsupported_extension_rejected(client):
    r = client.post(
        "/documents/ingest",
        files={"file": ("note.xyz", b"some content", "application/octet-stream")},
    )
    assert r.status_code == 400


def test_delete_document_removes_chunks(wired_singletons, client):
    body = b"# T\n\nfoo bar baz. " * 5
    r = client.post(
        "/documents/ingest",
        files={"file": ("a.md", body, "text/markdown")},
    )
    assert r.status_code == 200
    doc_id = r.json()["document_id"]
    chunks_before = wired_singletons["vectordb"].count()
    assert chunks_before >= 1

    r2 = client.delete(f"/documents/{doc_id}")
    assert r2.status_code == 200
    deleted = r2.json()["deleted_chunks"]
    assert deleted == chunks_before
    assert wired_singletons["vectordb"].count() == 0


def test_delete_nonexistent_document(wired_singletons, client):
    r = client.delete("/documents/does-not-exist")
    assert r.status_code == 200
    assert r.json()["deleted_chunks"] == 0
