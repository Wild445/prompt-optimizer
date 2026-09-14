"""Entry point for the local chat UI: `python run_ui.py`."""

from __future__ import annotations

import os
import webbrowser

import uvicorn

HOST = os.getenv("PROMPT_CREATOR_HOST", "127.0.0.1")
PORT = int(os.getenv("PROMPT_CREATOR_PORT", "8000"))

if __name__ == "__main__":
    if os.getenv("PROMPT_CREATOR_OPEN_BROWSER", "1") == "1":
        webbrowser.open(f"http://{HOST}:{PORT}")
    uvicorn.run("webapp.app:app", host=HOST, port=PORT, reload=False)
