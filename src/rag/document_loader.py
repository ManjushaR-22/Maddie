import os
import tempfile
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdf(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """Loads a PDF from disk and splits it into chunks."""
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
    except Exception:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        docs = loader.load()

    return split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

def load_and_split_text(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """Loads a text/markdown file from disk and splits it into chunks."""
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    return split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

def process_uploaded_file(
    file_bytes: bytes,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """Saves uploaded bytes to a temporary file, processes it, and returns split chunks."""
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        if suffix == ".pdf":
            chunks = load_and_split_pdf(tmp_path, chunk_size, chunk_overlap)
        else:
            chunks = load_and_split_text(tmp_path, chunk_size, chunk_overlap)

        # Update metadata to reflect actual filename rather than temporary path
        for chunk in chunks:
            chunk.metadata["source"] = filename
            chunk.metadata["file_name"] = filename

        return chunks
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """Splits documents into overlapping chunks with chunk indices."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks
