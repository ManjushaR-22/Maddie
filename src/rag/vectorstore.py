import os
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from pathlib import Path
from src.rag.config import VECTOR_STORE_DIR, DEFAULT_COLLECTION

class VectorStore:
    """Manages Chroma vector database storage and collections."""
    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        persist_directory: Optional[Path] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = str(persist_directory or VECTOR_STORE_DIR)
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self) -> None:
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Document embeddings for RAG system"}
        )

    def add_documents(self, documents: List[Any], embeddings: Any) -> int:
        """Adds LangChain Document objects and their embeddings to the vector store."""
        if len(documents) != len(embeddings):
            raise ValueError(f"Documents count ({len(documents)}) != embeddings count ({len(embeddings)})")

        if len(documents) == 0:
            return 0

        ids = []
        metadatas = []
        documents_texts = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(getattr(doc, "metadata", {}))
            metadata["chunk_index"] = i
            metadata["content_length"] = len(doc.page_content)
            # Ensure all metadata values are str, int, float, or bool for ChromaDB
            clean_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_metadata[k] = v
                else:
                    clean_metadata[k] = str(v)
            metadatas.append(clean_metadata)

            documents_texts.append(doc.page_content)
            if hasattr(embedding, "tolist"):
                embeddings_list.append(embedding.tolist())
            else:
                embeddings_list.append(list(embedding))

        self.collection.add(
            ids=ids,
            metadatas=metadatas,
            documents=documents_texts,
            embeddings=embeddings_list
        )
        return len(ids)

    def count(self) -> int:
        return self.collection.count() if self.collection else 0

    def list_collections(self) -> List[str]:
        return [c.name for c in self.client.list_collections()] if self.client else []

    def reset_collection(self) -> None:
        """Empties the current collection."""
        if self.client:
            self.client.delete_collection(self.collection_name)
            self._initialize_store()
