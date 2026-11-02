import os
import sys
import re
from pathlib import Path

# Ensure UTF-8 output encoding to avoid charmap / cp1252 errors on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import gdown
from gdown.download_folder import (
    _extract_folder_id,
    _get_session,
    _download_and_parse_google_drive_link,
    _get_directory_structure,
)

FOLDER_URL = "https://drive.google.com/drive/folders/1nMj3ndWG-NAXRrPypsBCsEVNSa48bys9"
DEST_DIR = Path("data/pdf")
DEST_DIR.mkdir(parents=True, exist_ok=True)

def sanitize_name(name: str) -> str:
    """Sanitize directory or file name for Windows filesystem."""
    # Replace illegal Windows characters: \ / : * ? " < > |
    sanitized = re.sub(r'[\\/:*?"<>|]', "_", name).strip()
    return sanitized if sanitized else "unnamed"

def collect_pdf_files(url: str):
    folder_id = _extract_folder_id(url=url)
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
        print("🔍 Scanning Google Drive folder structure (read-only)...")
        gdrive_file = _download_and_parse_google_drive_link(
            sess=sess,
            folder_id=folder_id,
            quiet=True,
            verify=True,
            timeout=10,
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
                pdf_files.append((item.id, item.name, rel_path))

    return pdf_files

def main():
    print("=" * 60)
    print("📥 Scanning Google Drive for PDF documents...")
    print(f"🔗 Source: {FOLDER_URL}")
    print(f"📂 Destination: {DEST_DIR.resolve()}")
    print("=" * 60)

    pdfs = collect_pdf_files(FOLDER_URL)
    print(f"\n✅ Discovered {len(pdfs)} PDF file(s) in Drive folder.\n")

    downloaded = 0
    skipped = 0
    failed = 0

    for idx, (file_id, file_name, rel_path) in enumerate(pdfs, 1):
        # Sanitize path parts
        path_parts = Path(rel_path).parts
        clean_parts = [sanitize_name(p) for p in path_parts]
        target_path = DEST_DIR.joinpath(*clean_parts)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists() and target_path.stat().st_size > 0:
            print(f"[{idx}/{len(pdfs)}] ⏩ Exists: {target_path.name}")
            skipped += 1
            continue

        print(f"[{idx}/{len(pdfs)}] ⬇️ Downloading: {target_path.name}")
        try:
            gdown.download(
                id=file_id,
                output=str(target_path),
                quiet=True,
            )
            if target_path.exists() and target_path.stat().st_size > 0:
                size_mb = target_path.stat().st_size / (1024 * 1024)
                print(f"   ✓ Saved {target_path.name} ({size_mb:.2f} MB)")
                downloaded += 1
            else:
                print(f"   ⚠️ File not saved or 0 bytes: {target_path.name}")
                failed += 1
        except Exception as e:
            print(f"   ❌ Error downloading {target_path.name}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"🎉 Download Summary: {downloaded} downloaded, {skipped} existing, {failed} failed out of {len(pdfs)} total.")
    print("=" * 60)

if __name__ == "__main__":
    main()
