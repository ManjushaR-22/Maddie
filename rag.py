import sys
import subprocess
from pathlib import Path

def main():
    """CLI entry point to run the Streamlit chatbot interface."""
    app_path = Path(__file__).resolve().parent / "app.py"
    print("=" * 60)
    print("🚀 Launching RAG Document Chatbot Web Interface...")
    print(f"📄 Application: {app_path}")
    print("=" * 60)

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Chatbot server stopped.")

if __name__ == "__main__":
    main()
