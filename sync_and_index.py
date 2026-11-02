import os
import sys
import json
import re
import time
from pathlib import Path

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import gdown
from gdown.download_folder import (
    _extract_folder_id,
    _get_session,
    _download_and_parse_google_drive_link,
    _get_directory_structure,
)
from src.rag.config import PDF_DIR
from src.rag.engine import RagEngine
from src.rag.document_loader import load_and_split_pdf

FOLDER_URL = "https://drive.google.com/drive/folders/1nMj3ndWG-NAXRrPypsBCsEVNSa48bys9"
MANIFEST_FILE = Path("data/drive_pdf_manifest.json")
DEST_DIR = PDF_DIR
DEST_DIR.mkdir(parents=True, exist_ok=True)

def sanitize_name(name: str) -> str:
    """Sanitize directory or file name for Windows filesystem."""
    sanitized = re.sub(r'[\\/:*?"<>|]', "_", name).strip()
    return sanitized if sanitized else "unnamed"

def get_already_indexed_sources(vector_store) -> set:
    """Retrieves the set of source file names already in the vector database."""
    try:
        data = vector_store.collection.get(include=["metadatas"])
        indexed = set()
        if data and data.get("metadatas"):
            for meta in data["metadatas"]:
                if meta:
                    src = meta.get("file_name") or meta.get("source")
                    if src:
                        indexed.add(Path(src).name)
        return indexed
    except Exception as e:
        print(f"Notice: Could not fetch indexed sources ({e}). Starting fresh.")
        return set()

def fetch_or_load_manifest(folder_url: str):
    """Loads manifest if cached, otherwise scans Google Drive and caches it."""
    if MANIFEST_FILE.exists():
        try:
            cached = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
            if cached and isinstance(cached, list) and len(cached) > 0:
                print(f"📋 Loaded {len(cached)} PDF entries from cached manifest.")
                return cached
        except Exception as e:
            print(f"Warning: Manifest cache unreadable ({e}), re-scanning Drive...")

    folder_id = _extract_folder_id(url=folder_url)
    folder_user_agent = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
    )
    sess, _ = _get_session(
        proxy=None,
        use_cookies=False,
        user_agent=folder_user_agent,
        cookies_file=None,
    )
    try:
        print("🔍 Crawling Google Drive folder structure (read-only)...")
        gdrive_file = _download_and_parse_google_drive_link(
            sess=sess,
            folder_id=folder_id,
            quiet=True,
            verify=True,
            timeout=15,
        )
    finally:
        sess.close()

    directory_structure = _get_directory_structure(
        gdrive_file=gdrive_file, previous_path=""
    )

    pdf_files = []
    for item, rel_path in directory_structure:
        if item is not None and item.name:
            if item.name.lower().endswith(".pdf"):
                pdf_files.append({
                    "id": item.id,
                    "name": item.name,
                    "rel_path": rel_path
                })

    MANIFEST_FILE.write_text(json.dumps(pdf_files, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Discovered and cached {len(pdf_files)} PDF entries to {MANIFEST_FILE}.")
    return pdf_files

def main():
    print("=" * 65)
    print("🩺 Meddie Knowledge Base Sync & Indexing Pipeline")
    print(f"🔗 Source Drive: {FOLDER_URL}")
    print(f"📂 PDF Directory: {DEST_DIR.resolve()}")
    print("=" * 65)

    engine = RagEngine()
    current_chunk_count = engine.vector_store.count()
    print(f"📊 Current ChromaDB Chunks: {current_chunk_count}")

    indexed_sources = get_already_indexed_sources(engine.vector_store)
    print(f"📚 Documents already indexed in DB: {len(indexed_sources)}")

    manifest = fetch_or_load_manifest(FOLDER_URL)
    total_files = len(manifest)

    downloaded_count = 0
    indexed_count = 0

    for idx, entry in enumerate(manifest, 1):
        file_id = entry["id"]
        file_name = entry["name"]
        rel_path = entry["rel_path"]

        # Build clean target path
        path_parts = Path(rel_path).parts
        clean_parts = [sanitize_name(p) for p in path_parts]
        target_path = DEST_DIR.joinpath(*clean_parts)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Download if missing
        if not target_path.exists() or target_path.stat().st_size == 0:
            print(f"[{idx}/{total_files}] ⬇️ Downloading: {target_path.name}")
            try:
                gdown.download(id=file_id, output=str(target_path), quiet=True)
                time.sleep(0.3)
                if target_path.exists() and target_path.stat().st_size > 0:
                    mb = target_path.stat().st_size / (1024 * 1024)
                    print(f"   ✓ Downloaded ({mb:.2f} MB)")
                    downloaded_count += 1
                else:
                    print(f"   ⚠️ Download yielded empty file, skipping.")
                    continue
            except Exception as e:
                print(f"   ❌ Download error: {e}")
                continue
        else:
            print(f"[{idx}/{total_files}] 📁 Local: {target_path.name}")

        # 2. Index into ChromaDB if not already indexed
        if target_path.name in indexed_sources:
            print(f"   ⏩ Already indexed in ChromaDB.")
            continue

        print(f"   🧠 Splitting & Indexing into ChromaDB...")
        try:
            chunks = load_and_split_pdf(str(target_path), chunk_size=1000, chunk_overlap=200)
            if not chunks:
                print(f"   ⚠️ No extractable text found (scanned images or empty).")
                indexed_sources.add(target_path.name)
                continue

            for c in chunks:
                c.metadata["file_name"] = target_path.name
                c.metadata["source"] = str(target_path.relative_to(DEST_DIR))

            chunk_texts = [c.page_content for c in chunks]
            embeddings = engine.embedding_manager.embed_texts(chunk_texts, batch_size=64, show_progress=False)
            added = engine.vector_store.add_documents(chunks, embeddings)
            indexed_sources.add(target_path.name)
            indexed_count += 1
            print(f"   ✅ Indexed {added} chunks. (Total Chunks: {engine.vector_store.count()})")
        except Exception as e:
            print(f"   ❌ Indexing error: {e}")

    print("\n" + "=" * 65)
    print(f"🎉 Sync Complete for Dr. Kallu's Meddie!")
    print(f"📦 Total PDFs in Manifest: {total_files}")
    print(f"⬇️ New Downloads: {downloaded_count}")
    print(f"📑 Newly Indexed: {indexed_count}")
    print(f"📊 Final Chunks in ChromaDB: {engine.vector_store.count()}")
    print("=" * 65)

if __name__ == "__main__":
    main()
