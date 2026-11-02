# 📚 RAG Document Intelligence Chatbot

An interactive conversational Retrieval-Augmented Generation (RAG) assistant powered by **LangChain**, **ChromaDB**, **Sentence Transformers**, and high-speed **Groq LLMs**.

---

## 🌟 Features

- **💬 Real-Time Streaming Chat**: Fast token-by-token streaming responses powered by Groq.
- **🔍 Grounded Source Citations**: Inspect retrieved chunk excerpts, similarity scores, ranks, and page numbers directly under each response.
- **📄 On-the-Fly Document Ingestion**: Upload `.pdf`, `.txt`, or `.md` files via the sidebar to immediately chunk, embed, and index them into ChromaDB.
- **🤖 Dynamic Model Switching**: Switch between active Groq models (`openai/gpt-oss-20b`, `qwen/qwen3.8-27b`, `openai/gpt-oss-120b`).
- **🧠 Conversational Memory**: Retains conversation history context across multiple dialogue turns.
- **🎛️ Tunable RAG Parameters**: Interactive sliders for Top-K retrieval chunks, Temperature, Chunk size, and Chunk overlap.

---

## 🚀 Quick Start

### 1. Ensure Environment Configuration
Check that your `.env` file in the project root contains your Groq API key:

```env
GROQ_API_KEY="gsk_..."
```
*(The system also recognizes `GROK_API_KEY` if already configured).*

### 2. Run the Chatbot Interface

You can start the web interface using any of the following commands:

Using `uv`:
```bash
uv run streamlit run app.py
```

Or using `python`:
```bash
python rag.py
```

The application will launch and open in your default browser at:
```
http://localhost:8501
```

---

## 📁 Project Architecture

```
rag/
├── app.py                     # Streamlit Interactive Chatbot Web Interface
├── rag.py                     # CLI launcher entrypoint
├── pyproject.toml             # Project dependencies and metadata
├── requirements.txt           # Standard requirements list
├── .env                       # Environment variables (API keys)
├── data/
│   ├── pdf/                   # Original PDF source files
│   ├── text_files/            # Text document storage
│   └── vector_store/          # Persistent ChromaDB vector database
├── notebook/
│   └── document.ipynb         # Original exploratory Jupyter notebook
└── src/
    └── rag/
        ├── __init__.py        # Package exports
        ├── config.py          # Central paths and model configuration
        ├── embeddings.py      # EmbeddingManager (SentenceTransformers)
        ├── vectorstore.py     # ChromaDB VectorStore management
        ├── document_loader.py # PDF/TXT loader and recursive chunker
        └── engine.py          # RagEngine with retrieval, streaming, & prompt synthesis
```

---

## ⚙️ Configuration & Customization

- **Embedding Model**: Default is `all-MiniLM-L6-v2` (configured in [`src/rag/config.py`](file:///c:/Users/HP/Desktop/rag/src/rag/config.py)).
- **LLM Models**: Choose from supported models in the UI sidebar dropdown or add more in `AVAILABLE_MODELS` inside [`src/rag/config.py`](file:///c:/Users/HP/Desktop/rag/src/rag/config.py).
