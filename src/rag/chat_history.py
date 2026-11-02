import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

HISTORY_FILE = Path("data/chat_history.json")

MEDDIE_WELCOME_MESSAGE = (
    "Hello Dr. Kallu! 🩺 What can I help you with today on your medical journey?\n\n"
    "Whether it's discussing complex clinical presentations, revising high-yield topics, "
    "or reviewing case sheets, I'm right here with you. Remember: every case you study and every "
    "hour you dedicate brings you one step closer to mastering clinical excellence. Let's make today count! 💪"
)

def _ensure_history_file():
    if not HISTORY_FILE.parent.exists():
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text("{}", encoding="utf-8")

def load_all_sessions() -> Dict[str, Dict[str, Any]]:
    """Loads all chat sessions from disk, ordered by most recently updated."""
    _ensure_history_file()
    try:
        content = HISTORY_FILE.read_text(encoding="utf-8").strip()
        if not content:
            return {}
        data = json.loads(content)
        # Sort sessions by updated_at descending
        sorted_sessions = dict(
            sorted(
                data.items(),
                key=lambda item: item[1].get("updated_at", ""),
                reverse=True
            )
        )
        return sorted_sessions
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return {}

def save_session(session_id: str, title: str, messages: List[Dict[str, Any]]) -> None:
    """Saves or updates a chat session."""
    _ensure_history_file()
    sessions = load_all_sessions()
    now_str = datetime.now().strftime("%b %d, %I:%M %p")

    # If title is default or empty, infer from first user message
    if not title or title.startswith("New Case") or title == "New Consultation":
        for m in messages:
            if m.get("role") == "user":
                user_text = m.get("content", "").strip()
                title = (user_text[:28] + "...") if len(user_text) > 28 else user_text
                break
        if not title:
            title = f"Case #{len(sessions) + 1}"

    sessions[session_id] = {
        "id": session_id,
        "title": title,
        "updated_at": now_str,
        "messages": messages,
    }

    try:
        HISTORY_FILE.write_text(json.dumps(sessions, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"Error saving chat history: {e}")

def create_new_session(custom_title: Optional[str] = None) -> tuple[str, Dict[str, Any]]:
    """Creates a new empty session with Meddie's personalized greeting to Dr. Kallu."""
    session_id = str(uuid.uuid4())[:8]
    title = custom_title or "New Consultation"
    initial_messages = [
        {
            "role": "assistant",
            "content": MEDDIE_WELCOME_MESSAGE,
            "sources": []
        }
    ]
    save_session(session_id, title, initial_messages)
    return session_id, {
        "id": session_id,
        "title": title,
        "updated_at": datetime.now().strftime("%b %d, %I:%M %p"),
        "messages": initial_messages
    }

def delete_session(session_id: str) -> None:
    """Deletes a session from history."""
    _ensure_history_file()
    sessions = load_all_sessions()
    if session_id in sessions:
        del sessions[session_id]
        HISTORY_FILE.write_text(json.dumps(sessions, indent=2, ensure_ascii=False), encoding="utf-8")
