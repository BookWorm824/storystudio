"""StoryStudio – your AI editing helper for stories, movies and music videos.

Checks GitHub for a newer version first (ss_update.py) and installs it, then opens the
StoryStudio window (a local page shown in Microsoft Edge's app mode). See ss_server.py for the
window side and storysound.py for the engine.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def update_first():
    """Install a newer version before anything else is loaded. Never stops StoryStudio from
    opening: no internet or a failed download just means it opens the version it has."""
    if os.environ.get("STORYSTUDIO_NO_UPDATE") or os.environ.get("STORYSTUDIO_NO_WINDOW"):
        return
    try:
        import ss_update
        info = ss_update.check(timeout=6)
        if not info:
            return
        win = None
        try:
            import tkinter as tk
            win = tk.Tk()
            win.title("StoryStudio")
            win.configure(bg="#12151c")
            win.geometry("420x120")
            win.resizable(False, False)
            tk.Label(win, text=f"Updating StoryStudio to version {info['version']}…",
                     fg="#eef0f4", bg="#12151c", font=("Segoe UI", 12)).pack(pady=(28, 6))
            msg = tk.Label(win, text="This takes a moment.", fg="#9aa3b2", bg="#12151c",
                           font=("Segoe UI", 10))
            msg.pack()
            win.update()

            def progress(f, m):
                msg.config(text=f"{int(f * 100)}%  {m}")
                win.update()
        except Exception:
            progress = None
        ss_update.apply(info, progress)
        if win is not None:
            win.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    update_first()
    import ss_server  # noqa: E402  (imported after the update, so new files are used)
    ss_server.main()
