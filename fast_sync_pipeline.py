import os
import sys
import json
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import gdown
from src.rag.config import PDF_DIR
from src.rag.engine import RagEngine
from src.rag.document_loader import load_and_split_pdf

MANIFEST_FILE = Path("data/drive_pdf_manifest.json")
PROGRESS_FILE = Path("data/sync_progress.json")
DEST_DIR = PDF_DIR
DEST_DIR.mkdir(parents=True, exist_ok=True)

TEXTBOOK_KEYWORDS = [
    "textbook", "book", "dadapeer", "ghai", "hutchison", "kundu", 
    "chaurasia", "davidson", "harrison", "bailey", "love", "parikh", 
    "dutta", "shaw", "park", "marwah", "apoorv", "tripathi", "snell", 
    "guyton", "robbins", "bansal", "wakode", "handbook", "manual", 
    "singi", "revision", "coaching"
]

def sanitize_name(name: str) -> str:
    sanitized = re.sub(r'[\\/:*?"<>|]', "_", name).strip()
    return sanitized if sanitized else "unnamed"

def is_textbook(entry: dict) -> bool:
    text = (entry.get("name", "") + " " + entry.get("rel_path", "")).lower()
    return any(k in text for k in TEXTBOOK_KEYWORDS)

def download_single_file(entry: dict) -> tuple[dict, bool, str]:
    file_id = entry["id"]
    rel_path = entry["rel_path"]
    path_parts = Path(rel_path).parts
    clean_parts = [sanitize_name(p) for p in path_parts]
    target_path = DEST_DIR.joinpath(*clean_parts)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and target_path.stat().st_size > 0:
        return entry, True, str(target_path)

    try:
        gdown.download(id=file_id, output=str(target_path), quiet=True)
        if target_path.exists() and target_path.stat().st_size > 0:
            return entry, True, str(target_path)
    except Exception as e:
        return entry, False, str(e)
    return entry, False, "Empty file"

def get_already_indexed_sources(vector_store) -> set:
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

def index_file(pdf_path: Path, engine: RagEngine, indexed_sources: set) -> int:
    if pdf_path.name in indexed_sources:
        return 0

    try:
        chunks = load_and_split_pdf(str(pdf_path), chunk_size=1000, chunk_overlap=200)
        if not chunks:
            indexed_sources.add(pdf_path.name)
            return 0

        for c in chunks:
            c.metadata["file_name"] = pdf_path.name
            c.metadata["source"] = str(pdf_path.relative_to(DEST_DIR))

        chunk_texts = [c.page_content for c in chunks]
        embeddings = engine.embedding_manager.embed_texts(chunk_texts, batch_size=64, show_progress=False)
        added = engine.vector_store.add_documents(chunks, embeddings)
        indexed_sources.add(pdf_path.name)
        return added
    except Exception as e:
        print(f"   ⚠️ Error indexing {pdf_path.name}: {e}")
        indexed_sources.add(pdf_path.name)
        return 0

def run_priority_sync():
    print("=" * 65)
    print("🚀 FAST MEDDIE SYNC & INDEXING PIPELINE")
    print(f"📂 PDFs Destination: {DEST_DIR.resolve()}")
    print("=" * 65)

    if not MANIFEST_FILE.exists():
        print("❌ Manifest file not found! Please run scan first.")
        return

    all_entries = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    print(f"📋 Loaded {len(all_entries)} total files from manifest.")

    engine = RagEngine()
    indexed_sources = get_already_indexed_sources(engine.vector_store)
    print(f"📊 Starting DB Chunks: {engine.vector_store.count()} | Already Indexed Files: {len(indexed_sources)}")

    # Split into Priority 1 (Textbooks) and Priority 2 (Others)
    textbooks = [e for e in all_entries if is_textbook(e)]
    others = [e for e in all_entries if not is_textbook(e)]
    print(f"⭐ Textbooks & Major Clinical References (Priority 1): {len(textbooks)} files")
    print(f"📄 Case Presentations & Clinical Notes (Priority 2): {len(others)} files\n")

    def process_group(group_entries, group_name):
        print(f"\n{'='*25} STARTING {group_name} ({len(group_entries)} files) {'='*25}")
        
        # Parallel download in batches
        max_workers = 4
        total = len(group_entries)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_entry = {executor.submit(download_single_file, e): e for e in group_entries}
            
            completed = 0
            for future in as_completed(future_to_entry):
                completed += 1
                entry, success, path_or_err = future.result()
                if success:
                    target_path = Path(path_or_err)
                    size_mb = target_path.stat().st_size / (1024 * 1024)
                    print(f"[{completed}/{total}] ⬇️ Downloaded: {target_path.name} ({size_mb:.1f} MB)")
                    
                    # Immediately index
                    if target_path.name not in indexed_sources:
                        added = index_file(target_path, engine, indexed_sources)
                        if added > 0:
                            print(f"   🧠 Indexed {added} chunks -> Total DB: {engine.vector_store.count()}")
                else:
                    print(f"[{completed}/{total}] ❌ Failed {entry['name']}: {path_or_err}")

        print(f"\n✅ {group_name} FULLY PROCESSED! (Total Chunks now: {engine.vector_store.count()})")

    # PHASE 1: TEXTBOOKS
    process_group(textbooks, "PHASE 1: TEXTBOOKS & MAJOR REFERENCES")
    
    # Textbooks complete notification banner in logs
    print("\n" + "#" * 65)
    print("📢 NOTIFICATION: ALL TEXTBOOKS & MAJOR REFERENCES COMPLETED!")
    print(f"📈 Current Total Chunks in ChromaDB: {engine.vector_store.count()}")
    print("#" * 65 + "\n")

    # PHASE 2: REMAINING FILES
    process_group(others, "PHASE 2: ALL REMAINING CLINICAL FILES")

    print("\n" + "=" * 65)
    print("🎉 ALL FILES (TEXTBOOKS + ALL CLINICAL DATA) FULLY SYNCED & INDEXED!")
    print(f"🏆 Final Total Chunks in ChromaDB: {engine.vector_store.count()}")
    print("=" * 65)

if __name__ == "__main__":
    run_priority_sync()
