from typing import List, Dict, Any, Optional, Generator
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.rag.embeddings import EmbeddingManager
from src.rag.vectorstore import VectorStore
from src.rag.config import get_groq_api_key, DEFAULT_LLM_MODEL

class RagEngine:
    """Core RAG retrieval and generation engine customized for Meddie & Dr. Kallu."""
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
    ):
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.vector_store = vector_store or VectorStore()
        self.api_key = get_groq_api_key()

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves top_k chunks from the Chroma vector store for a given query."""
        if self.vector_store.count() == 0:
            return []

        query_embedding = self.embedding_manager.embed_query(query)
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.vector_store.count()),
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"Chroma retrieval error: {e}")
            return []

        retrieved = []
        if results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            ids = results.get("ids", [[]])[0]

            for i, (doc_id, doc_text, meta, dist) in enumerate(zip(ids, docs, metadatas, distances)):
                score = round(1.0 - (dist if dist is not None else 0.0), 3)
                if score_threshold is not None and score < score_threshold:
                    continue

                retrieved.append({
                    "id": doc_id,
                    "content": doc_text,
                    "metadata": meta or {},
                    "distance": dist,
                    "score": score,
                    "rank": i + 1,
                })
        return retrieved

    def _build_messages(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> list:
        """Constructs system instruction, conversation history, and current question."""
        if context_docs:
            context_blocks = []
            for doc in context_docs:
                src = doc.get("metadata", {}).get("source") or doc.get("metadata", {}).get("file_name", "Medical Document")
                page = doc.get("metadata", {}).get("page")
                page_str = f" (Page {page})" if page is not None else ""
                context_blocks.append(f"--- SOURCE: {src}{page_str} ---\n{doc['content']}")
            formatted_context = "\n\n".join(context_blocks)
            doc_context_section = f"=== CLINICAL & DOCUMENT CONTEXT ===\n{formatted_context}\n==================================="
        else:
            doc_context_section = (
                "=== CLINICAL CONTEXT ===\n"
                "No direct context chunks found for this query in the indexed documents. "
                "Draw upon standard medical knowledge while noting that the uploaded files didn't have a direct excerpt.\n"
                "========================"
            )

        system_instruction = (
            "You are **Meddie** 🩺, an intelligent, empathetic, and uplifting AI medical mentor and study partner for **Dr. Kallu**.\n\n"
            "Key Instructions:\n"
            "1. **Address User**: Always address the user warmly and respectfully as **Dr. Kallu**.\n"
            "2. **Motivational Role**: You are Dr. Kallu's biggest supporter in their medical journey! Medical studies, clinical postings, and exams are demanding; frequently offer brief, genuine words of encouragement, resilience, and clinical motivation (e.g., 'You're doing fantastic, Dr. Kallu!', 'Mastering this will make you an even sharper physician!', 'Keep going, Dr. Kallu—every case makes a difference!').\n"
            "3. **Clinical Accuracy & Structure**: Provide clear, high-yield medical explanations, organized with bullet points, headings, differential diagnoses, clinical presentations, or management guidelines when appropriate.\n"
            "4. **Grounded in Document Context**: Base your specific answers on the provided document excerpts whenever relevant and cite the source note or textbook name.\n\n"
            f"{doc_context_section}"
        )

        messages = [SystemMessage(content=system_instruction)]

        # Append recent conversation history (last 6 turns)
        if chat_history:
            recent_history = chat_history[-6:]
            for turn in recent_history:
                role = turn.get("role")
                content = turn.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=query))
        return messages

    def generate_answer_stream(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
        model_name: str = DEFAULT_LLM_MODEL,
        temperature: float = 0.3
    ) -> Generator[str, None, None]:
        """Streams answer tokens generator for real-time typing effect."""
        llm = ChatGroq(
            model=model_name,
            groq_api_key=self.api_key,
            temperature=temperature,
            streaming=True
        )
        messages = self._build_messages(query, context_docs, chat_history)
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content
