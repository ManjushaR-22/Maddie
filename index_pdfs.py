import os
import sys
from pathlib import Path

# Ensure UTF-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.rag.config import PDF_DIR
from src.rag.engine import RagEngine
from src.rag.document_loader import load_and_split_pdf

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
        print(f"Notice: Could not fetch existing indexed sources ({e}). Starting fresh check.")
        return set()

def index_all_pdfs(batch_limit: int = None):
    engine = RagEngine()
    print("=" * 60)
    print("📚 Scanning data/pdf for PDF files to index into ChromaDB...")
    print(f"📊 Current chunks in database: {engine.vector_store.count()}")
    print("=" * 60)

    pdf_files = list(PDF_DIR.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF file(s) on disk.\n")

    if not pdf_files:
        print("No PDF files found to index.")
        return

    indexed_sources = get_already_indexed_sources(engine.vector_store)
    print(f"Previously indexed files in DB: {len(indexed_sources)}")

    to_index = [p for p in pdf_files if p.name not in indexed_sources]
    print(f"Files remaining to index: {len(to_index)}")

    if batch_limit:
        to_index = to_index[:batch_limit]
        print(f"Processing next batch of {len(to_index)} files...")

    total_chunks_added = 0

    for idx, pdf_path in enumerate(to_index, 1):
        print(f"\n[{idx}/{len(to_index)}] 📄 Processing: {pdf_path.name}")
        try:
            chunks = load_and_split_pdf(str(pdf_path), chunk_size=1000, chunk_overlap=200)
            if not chunks:
                print(f"   ⚠️ No text extractable from {pdf_path.name} (may be scanned images)")
                continue

            # Set clean metadata
            for c in chunks:
                c.metadata["file_name"] = pdf_path.name
                c.metadata["source"] = str(pdf_path.relative_to(PDF_DIR))

            print(f"   Splitted into {len(chunks)} chunks. Generating embeddings...")
            chunk_texts = [c.page_content for c in chunks]
            embeddings = engine.embedding_manager.embed_texts(chunk_texts, batch_size=64, show_progress=False)

            added = engine.vector_store.add_documents(chunks, embeddings)
            total_chunks_added += added
            print(f"   ✅ Added {added} chunks. (Total DB count: {engine.vector_store.count()})")

        except Exception as e:
            print(f"   ❌ Error indexing {pdf_path.name}: {e}")

    print("\n" + "=" * 60)
    print(f"🎉 Indexing Complete! Added {total_chunks_added} chunks.")
    print(f"📈 Total chunks in ChromaDB: {engine.vector_store.count()}")
    print("=" * 60)

if __name__ == "__main__":
    index_all_pdfs()
