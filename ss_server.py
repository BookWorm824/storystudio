"""StoryStudio's window: a small local web server that the app window (Microsoft Edge in app
mode) talks to. All the real work is done by the engine (storysound.py and friends); this
file only turns button presses into engine calls and reports progress back.

Only listens on 127.0.0.1, on a random port, and only serves media files from folders
StoryStudio itself uses.
"""
import json
import mimetypes
import os
import re
import secrets
import subprocess
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import ss_core as core
from ss_core import APP_DATA, APP_DIR, StoryError, fmt_time, log, log_error

UI_FILE = APP_DIR / "ss_ui.html"
TOKEN = secrets.token_hex(16)            # the window must send this (blocks other web pages)

KIND_NAMES = {"ambience": "Background", "footsteps": "Footsteps", "action": "Action",
              "impact": "Bump / hit", "whoosh": "Whoosh", "animal": "Animal",
              "nature": "Nature", "weather": "Weather", "water": "Water",
              "vehicle": "Vehicle", "household": "Household", "magic": "Magic",
              "cartoon": "Cartoon", "crowd": "Crowd", "music": "Music", "other": "Other"}


# ---------------------------------------------------------------------------
# app state
# ---------------------------------------------------------------------------
class State:
    def __init__(self):
        self.settings = core.load_settings()
        self.st = None                  # the engine's run state (dict)
        self.job = None                 # current background job
        self.sb = None                  # the last shot list from the script builder
        self.asking = 0                 # slow AI questions being answered right now
        self.restart = False            # set after an update: start the new version
        self.last_ping = time.time()
        self.lock = threading.Lock()


S = State()
SLOW_ASKS = {"sb_redo", "sb_brainstorm", "sb_cast"}
PICK_LOCK = threading.Lock()            # one Windows file window at a time
# never wait behind other actions: quick status calls, and file pickers (a Windows file window
# left open must not freeze the rest of StoryStudio)
NO_LOCK = {"job", "ping", "stop", "client_error", "update_check", "sb_pick_text", "sb_pick_song",
           "pick_lyrics", "sb_open", "sb_result", "sb_styles", "sb_style_pics"}


def recent_add(path):
    rec = [p for p in S.settings.get("recent", []) if p != path]
    S.settings["recent"] = ([path] + rec)[:8]
    core.save_settings(S.settings)


def media_url(path):
    if not path:
        return ""
    return f"/media?t={TOKEN}&p={quote(str(path))}"


def allowed_media(path):
    """Only files StoryStudio made or works with: its own folders, and files of the open
    project (its folder and the StoryStudio/StorySound output folders)."""
    p = Path(path).resolve()
    roots = [APP_DATA]
    if S.st:
        proj = Path(S.st["project"]).parent
        roots += [proj / "StoryStudio", proj / "StorySound", proj.parent]
    return any(str(p).lower().startswith(str(r.resolve()).lower()) for r in roots)


def project_view():
    st = S.st
    if not st:
        return None
    i = st["info"]
    return {"name": Path(st["project"]).stem, "file": st["project"],
            "seq": st["seq_name"], "duration": fmt_time(st["duration"]),
            "seconds": st["duration"], "shots": i["shots"],
            "narration": i["dialogue_clips"], "missing": i["missing"],
            "export": Path(st["export"]).name if st.get("export") else "",
            "story": st.get("story") or "", "snapshot_time": st.get("snapshot_time", ""),
            "sequences": [{"uid": s["uid"], "name": s["name"],
                           "duration": fmt_time(s["duration"])} for s in st["sequences"]],
            "seq_uid": st["seq_uid"], "thumb": media_url(st.get("thumb")),
            "previous": st.get("previous", {}).get("stage") in ("sounds", "finished",
                                                                  "balanced") or
            (st.get("previous", {}).get("cues", 0) > 0 and
             st.get("previous", {}).get("stage") in ("karaoke", "feedback")),
            "previous_feedback": bool(st.get("previous", {}).get("feedback"))}


def shot_lists():
    try:
        import ss_script as SB
        return SB.recent(5)
    except Exception:
        log_error("shot lists")
        return []


def missing_tools():
    """Free tools StoryStudio downloads for itself the first time (no other app needed)."""
    import ss_media as M
    need = []
    if not core._find_tool("ffmpeg"):
        need.append("ffmpeg")
    if not M.speech_model_dir():
        need.append("speech")
    return need


def ai_view():
    import ss_claude as C
    import ss_gpt as G
    p = C.provider()
    return {"name": {"claude": "Claude", "chatgpt": "ChatGPT"}.get(p, ""), "provider": p or "",
            "claude": bool(C.find_claude()), "codex": bool(G.find()),
            "choice": S.settings.get("ai", "auto")}


FB_LABEL = {"fix": "Please fix", "suggest": "Idea", "beat": "Beat", "good": "Nicely done"}


def note_view(n):
    return {"id": n["id"], "t": n["time"], "time": fmt_time(n["time"]),
            "end": n.get("end"), "severity": n["severity"],
            "label": FB_LABEL.get(n["severity"], ""), "kind": n.get("kind", ""),
            "title": n["title"], "note": n["note"], "how": n.get("how", ""),
            "keep": bool(n.get("keep", True)), "ai": n.get("source") in ("ai", "claude")}


def feedback_view():
    st = S.st
    fb = (st or {}).get("feedback")
    if not fb:
        return None
    out = st.get("outputs") or {}
    return {"title": st["seq_name"], "kind": fb["kind"], "summary": fb["summary"],
            "by": fb.get("by", "Claude"),
            "stats": fb["stats"], "notes": [note_view(n) for n in fb["notes"]],
            "video": media_url(st.get("feedback_preview") or st.get("picture")),
            "project": out.get("project", ""), "project_error": out.get("project_error", ""),
            "folder": out.get("folder", ""), "list": out.get("list", ""),
            "dirty": bool(st.get("feedback_dirty")),
            "snapshot_time": st.get("snapshot_time", "")}


def cue_view(c):
    return {"id": c["id"], "time": fmt_time(c.get("place_at", c["start"])),
            "end": fmt_time(c["end"]), "what": c["what"],
            "kind": KIND_NAMES.get(c["kind"], c["kind"]), "kind_id": c["kind"],
            "keep": bool(c.get("keep")), "made": bool(c.get("file")),
            "source": {"pc": "Made on this PC", "freesound": "Freesound",
                       "adobe": "Not made"}.get(c.get("source", ""), ""),
            "audio": media_url(c.get("file")), "prompt": c.get("sound_prompt", ""),
            "adobe": c.get("adobe_prompt", ""), "suno": c.get("suno_prompt", ""),
            "note": c.get("note", ""), "credit": c.get("credit", ""),
            "volume": c.get("gain_extra", 0), "added": bool(c.get("added_by_check"))}


def review_view():
    st = S.st
    if not st:
        return None
    return {"title": st["seq_name"], "notes": st.get("plan_notes", ""),
            "notice": st.get("notice", ""), "levels": st.get("levels", []),
            "cues": [cue_view(c) for c in st.get("cues", [])]}


def outputs_view():
    st = S.st
    out = (st or {}).get("outputs") or {}
    return {"project": out.get("project", ""), "project_error": out.get("project_error", ""),
            "preview": media_url(out.get("preview")), "preview_file": out.get("preview", ""),
            "folder": out.get("folder", ""), "list": out.get("list", ""),
            "levels": (st or {}).get("levels", []), "mode": (st or {}).get("mode", ""),
            "snapshot_time": (st or {}).get("snapshot_time", ""),
            "overlay": out.get("overlay", ""), "lyrics": (st or {}).get("lyrics_text", ""),
            "source": (st or {}).get("project", ""), "seq_uid": (st or {}).get("seq_uid", "")}


# ---------------------------------------------------------------------------
# background jobs with progress and time left
# ---------------------------------------------------------------------------
class Job:
    def __init__(self, title, phases, then):
        self.title, self.then = title, then
        self.w = list(phases or [60.0])
        self.i, self.f, self.msg = 0, 0.0, "Starting…"
        self.t0 = self.pt0 = time.time()
        self.done = self.failed = False
        self.error, self.stop = "", False
        self.steps = []                 # finished step messages, for the list on screen

    def phase(self, i, seconds=None):
        self.i, self.f, self.pt0 = i, 0.0, time.time()
        if seconds is not None:
            self.w[i] = float(seconds)

    def progress(self, f, m):
        self.f = max(0.0, min(1.0, f))
        head = (m or "").split(":")[0].split("(")[0].rstrip("… .")
        if self.steps and self.steps[-1] != head and head:
            pass
        if head and (not self.steps or self.steps[-1] != head):
            self.steps.append(head)
            self.steps = self.steps[-8:]
        self.msg = m

    def overall(self):
        done = sum(self.w[:self.i]) + self.w[self.i] * self.f
        return done / max(1.0, sum(self.w))

    def left(self):
        spent = time.time() - self.pt0
        est = self.w[self.i] * (1 - self.f)
        if self.f > 0.08 and spent > 20:
            est = 0.3 * est + 0.7 * spent * (1 - self.f) / self.f
        return est + sum(self.w[self.i + 1:])

    def view(self):
        import storysound as E
        return {"title": self.title, "frac": self.overall(), "msg": self.msg,
                "left": E.nice_minutes(self.left()).capitalize() + " left",
                "steps": self.steps, "done": self.done, "failed": self.failed,
                "error": self.error, "then": self.then,
                "elapsed": int(time.time() - self.t0)}


def run_job(title, work, then, phases):
    if S.job and not S.job.done:
        raise StoryError("StoryStudio is already working on something.")
    job = Job(title, phases, then)
    S.job = job

    def go():
        try:
            work(job.progress, lambda: job.stop, job.phase)
        except StoryError as ex:
            job.failed, job.error = True, str(ex)
        except Exception:
            log_error("job " + title)
            job.failed = True
            job.error = ("Something went wrong. Details were saved to the log file in the "
                         "StoryStudio folder.")
        finally:
            job.done = True
            try:
                import storysound as E
                if S.st:
                    E.save(S.st)
            except Exception:
                pass
    threading.Thread(target=go, daemon=True).start()
    return job.view()


# ---------------------------------------------------------------------------
# actions the window can ask for
# ---------------------------------------------------------------------------
def choices(opts=None):
    s = dict(S.settings)
    s.update({k: v for k, v in (opts or {}).items() if k in core.DEFAULTS or k in (
        "best_of", "hf_key", "add_music")})
    return s


def native_pick(title, types, initial=None):
    """A normal Windows file picker (the browser window can't give real file paths)."""
    import tkinter as tk
    from tkinter import filedialog
    if not PICK_LOCK.acquire(blocking=False):
        raise StoryError("A file window is already open. Choose a file there (it may be "
                         "behind this window), or close it, then try again.")
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.lift()
        root.focus_force()
        try:
            f = filedialog.askopenfilename(title=title, filetypes=types,
                                           initialdir=initial or core.documents_dir())
        finally:
            root.destroy()
    finally:
        PICK_LOCK.release()
    return f


def make_thumb(st):
    """A frame from the middle of the movie, for the project card."""
    try:
        import storysound as E
        import ss_premiere as P
        seq = E._sequence(st)
        segs = [s for s in P.picture_segments(seq) if s[2] is not None and
                Path(s[2]["file"]).suffix.lower() not in P.IMAGE_EXTS]
        if not segs:
            return ""
        a, b, it = segs[len(segs) // 3]
        t = it["src_in"] + (b - a) / 2 * it["speed"]
        out = Path(st["work"]) / "thumb.jpg"
        core.run([core.ffmpeg(), "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", it["file"],
                  "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "4", out])
        return str(out)
    except Exception:
        log_error("thumb")
        return ""


def api(name, a):
    import storysound as E
    if name == "hello":
        return {"settings": {k: S.settings.get(k) for k in (
                    "busy", "add_music", "use_local", "use_freesound", "best_of")},
                "has_fs_key": bool(S.settings.get("freesound_key")),
                "has_hf_key": bool(S.settings.get("hf_key")),
                "recent": [{"path": p, "name": Path(p).stem, "folder": Path(p).parent.name}
                           for p in S.settings.get("recent", []) if os.path.exists(p)],
                "project": project_view(), "ai": ai_view(), "needs": missing_tools(),
                "shotlists": shot_lists(),
                "job": S.job.view() if S.job and not S.job.done else None}
    if name == "setup_tools":
        def work(progress, cancel, phase):
            need = missing_tools()
            if "ffmpeg" in need:
                progress(0.0, "Downloading ffmpeg, the free tool that reads video…")
                core.download_ffmpeg(lambda f: progress(
                    0.2 * f, "Downloading ffmpeg, the free tool that reads video…"))
            if "speech" in need:
                import ss_media as M
                progress(0.2, "Downloading the speech model that hears the words…")
                M.download_speech_model(lambda f: progress(
                    0.2 + 0.8 * f, "Downloading the speech model that hears the words…"))
            progress(1.0, "StoryStudio is ready.")
        return {"job": run_job("Getting StoryStudio ready", work, "home", [240])}
    if name == "ai_status":
        import ss_gpt as G
        v = ai_view()
        v["chatgpt"] = G.status()
        return {"ai": v}
    if name == "ai_choose":
        S.settings["ai"] = a.get("choice", "auto")
        core.save_settings(S.settings)
        return {"ai": ai_view()}
    if name == "ai_install_chatgpt":
        import ss_gpt as G
        return {"job": run_job("Setting up ChatGPT", lambda p, c, ph: G.install(p), "home",
                               [90])}
    if name == "ai_sign_in_chatgpt":
        import ss_gpt as G
        G.open_sign_in()
        return {"ok": True}
    if name == "pick_project":
        f = native_pick("Choose your Premiere project", [("Premiere projects", "*.prproj")],
                        S.settings.get("last_folder"))
        if not f:
            return {"project": project_view()}
        a = {"path": f}
        name = "open_project"
    if name == "open_project":
        path = a["path"]
        S.settings["last_folder"] = str(Path(path).parent)
        S.st = E.open_project(path, a.get("seq_uid"),
                              a.get("snap") if a.get("seq_uid") and S.st else None)
        S.st["thumb"] = make_thumb(S.st)
        recent_add(str(Path(path).resolve()))
        return {"project": project_view()}
    if name == "pick_story":
        f = native_pick("Choose your story or script",
                        [("Story or script", "*.docx *.txt *.md *.rtf *.pdf"), ("All", "*.*")],
                        str(Path(S.st["project"]).parent) if S.st else None)
        if f and S.st:
            S.st["story"] = f
        return {"project": project_view()}
    if name == "estimate":
        if not S.st:
            return {}
        e = E.estimate(S.st, choices(a), gpu_gb=a.get("gpu"))
        total = e["prepare"] + e["plan"] + e["sounds"] + e["finish"]
        return {"text": E.nice_minutes(total), "sounds": e["n"], "busy_gpu": e["busy_gpu"]}
    if name == "save_settings":
        for k in ("busy", "add_music", "use_local", "use_freesound", "best_of"):
            if k in a:
                S.settings[k] = a[k]
        for k in ("freesound_key", "hf_key"):
            if a.get(k) is not None and a.get(k) != "":
                S.settings[k] = a[k].strip()
        core.save_settings(S.settings)
        return {"ok": True}
    if name == "start_sounds":
        api("save_settings", a)
        st, s = S.st, choices(a)
        est = E.estimate(st, s)

        def work(progress, cancel, phase):
            phase(0)
            E.analyze(st, s, progress, cancel)
            phase(1, E.estimate(st, s, n_cues=len(st["cues"]))["sounds"])
            E.make_sounds(st, s, progress, cancel)
        return {"job": run_job("Finding sounds for your movie", work, "review",
                               [est["prepare"] + est["plan"], est["sounds"]])}
    if name == "start_fix_audio":
        api("save_settings", a)
        st, s = S.st, choices(a)
        e = E.estimate(st, s, n_cues=0)
        return {"job": run_job("Fixing the audio",
                               lambda p, c, ph: E.fix_audio(st, s, p, c), "done",
                               [e["prepare"] + e["finish"] + 60])}
    if name == "pick_lyrics":
        f = native_pick("Choose the lyrics", [("Lyrics", "*.txt *.docx *.md *.rtf *.pdf"),
                                              ("All", "*.*")],
                        str(Path(S.st["project"]).parent) if S.st else None)
        return {"text": core.read_story(f) if f else ""}
    if name == "start_karaoke":
        import ss_karaoke as K
        st, s = S.st, choices()
        text = a.get("lyrics", "")
        if not K.parse_lyrics(text):
            raise StoryError("Please paste the song's lyrics first.")
        st["lyrics_text"] = text
        est = 40 + st["duration"] * 0.6 + (0 if st.get("picture") else 30)
        return {"job": run_job("Making karaoke lyrics",
                               lambda p, c, ph: K.karaoke(st, s, text, p, c), "done", [est])}
    # --- script builder (no Premiere project needed) ---------------------------------
    if name == "sb_styles":
        import ss_script as SB
        return {"styles": SB.styles()["styles"], "aspects": list(SB.ASPECTS),
                "recent": SB.recent()}
    if name == "sb_style_pics":
        f = Path(__file__).with_name("ss_style_pics.json")
        try:
            return {"pics": json.loads(f.read_text(encoding="utf8"))}
        except (OSError, ValueError):
            return {"pics": {}}
    if name == "sb_save_style":
        import ss_script as SB
        sid = SB.save_style(a.get("name"), a.get("style"), a.get("clean", ""))
        return {"id": sid, "styles": SB.styles()["styles"]}
    if name == "sb_delete_style":
        import ss_script as SB
        SB.delete_style(a.get("id", ""))
        return {"styles": SB.styles()["styles"]}
    if name == "sb_pick_text":
        story = a.get("kind") == "story"
        f = native_pick("Choose the story script" if story else "Choose the lyrics",
                        [("Scripts and lyrics", "*.docx *.txt *.md *.rtf *.pdf"),
                         ("All", "*.*")], S.settings.get("last_folder"))
        if not f:
            return {"text": ""}
        S.settings["last_folder"] = str(Path(f).parent)
        return {"text": core.read_story(f), "name": Path(f).stem}
    if name == "sb_pick_song":
        f = native_pick("Choose the song", [("Songs", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg"),
                                            ("All", "*.*")], S.settings.get("last_folder"))
        if not f:
            return {"seconds": 0}
        S.settings["last_folder"] = str(Path(f).parent)
        return {"seconds": round(float(core.probe(f).get("duration") or 0), 1),
                "name": Path(f).stem}
    if name == "sb_brainstorm":
        import ss_script as SB
        return {"ideas": SB.brainstorm(a.get("text", ""), a.get("notes", ""),
                                       a.get("style_name", ""))}
    if name == "sb_cast":
        import ss_script as SB
        return {"characters": SB.suggest_cast(a)}
    if name == "sb_start":
        import math
        import ss_script as SB
        spec = dict(a)
        if spec.get("kind") == "story":
            secs = len((spec.get("text") or "").split()) / SB.WORDS_PER_SEC
            clip = 8
        else:
            secs, clip = float(spec.get("song_seconds") or 0), int(spec.get("clip") or 8)
        shots = secs / clip * (1 + SB.EXTRA)
        est = 70 + 65 * math.ceil(max(1, shots) / 24)

        def work(progress, cancel, phase):
            S.sb = SB.build(spec, progress, cancel)
        return {"job": run_job("Planning your shots", work, "script", [est])}
    if name == "sb_result":
        return {"result": S.sb}
    if name == "sb_redo":
        import ss_script as SB
        if not S.sb:
            raise StoryError("The shot list isn't open any more. Open it again from the "
                             "script builder, then try once more.")
        which, n, note = a.get("which", "picture"), a.get("n", 0), a.get("note", "")
        log("script builder: new version", which, n, repr(note)[:200])
        if which == "character":                   # also updates every shot they're in
            r = S.sb
            c = r["characters"][int(n)]
            k = sum(1 for s in r["shots"] if c["name"] in s.get("characters", []))

            def work(progress, cancel, phase):
                progress(0.05, f"Writing a new look for {c['name']}…")
                SB.redo(r, n, which, note, progress)
            return {"job": run_job(f"Updating {c['name']}", work, "script",
                                   [25 + 25 * (1 + k // 12)])}
        out = SB.redo(S.sb, n, which, note)
        log("script builder: new version done", which, n)
        return out
    if name == "sb_open":
        import ss_script as SB
        log("script builder: open", a.get("path", ""))
        try:
            S.sb = SB.load(a["path"])
        except (OSError, ValueError, KeyError):
            log_error("sb_open")
            raise StoryError("That shot list couldn't be opened. It may have been moved or "
                             "renamed.")
        return {"result": S.sb}
    if name == "update_check":
        import ss_update
        info = ss_update.check(timeout=8)
        return {"update": {"version": info["version"], "notes": info.get("notes", "")}
                if info else None, "current": ss_update.local_version()}
    if name == "update_now":
        import ss_update
        if S.job and not S.job.done:
            raise StoryError("StoryStudio is busy. Update when it has finished.")
        info = ss_update.check(timeout=10)
        if not info:
            return {"ok": False, "message": "You already have the newest version."}
        if not ss_update.apply(info):
            raise StoryError("The update couldn't be downloaded. Check the internet and try "
                             "again; your StoryStudio still works as it is.")
        log("updated to", info["version"], "- restarting")
        S.restart = True
        return {"ok": True, "version": info["version"]}
    if name == "client_error":
        log("window problem:", str(a.get("msg", ""))[:3000])
        return {"ok": True}
    if name == "start_feedback":
        import math
        import ss_feedback as F
        st, s = S.st, choices()
        kind = a.get("kind") or "auto"
        use_claude = bool(a.get("claude", True))
        dur = st["duration"]
        est = 25 + dur * 0.12 + (0 if st.get("picture") or st.get("export") else dur * 0.35)
        if use_claude:
            est += 20 + 75 * math.ceil(dur / (160 if dur <= 420 else 320))
        st["feedback_dirty"] = False
        return {"job": run_job("Watching your edit",
                               lambda p, c, ph: F.feedback(st, s, p, c, kind, use_claude),
                               "feedback", [est])}
    if name == "feedback":
        return {"feedback": feedback_view()}
    if name == "open_previous_feedback":
        prev = E.load(S.st["work"])
        if not prev.get("feedback"):
            raise StoryError("There's no feedback from last time for this project.")
        prev["thumb"] = S.st.get("thumb")
        S.st = prev
        return {"feedback": feedback_view()}
    if name == "fb_toggle":
        n = next(x for x in S.st["feedback"]["notes"] if x["id"] == a["id"])
        n["keep"] = bool(a.get("keep", not n.get("keep", True)))
        S.st["feedback_dirty"] = True
        E.save(S.st)
        return {"note": note_view(n), "dirty": True}
    if name == "fb_save":
        import ss_feedback as F
        F.write_outputs(S.st)
        S.st["feedback_dirty"] = False
        E.save(S.st)
        return {"feedback": feedback_view()}
    if name == "open_previous":
        prev = E.load(S.st["work"])
        prev["story"] = S.st.get("story") or prev.get("story")
        prev["thumb"] = S.st.get("thumb")
        S.st = prev
        return {"review": review_view()}
    if name == "review":
        return {"review": review_view()}
    if name == "job":
        return {"job": S.job.view() if S.job else None}
    if name == "stop":
        if S.job:
            S.job.stop = True
        return {"ok": True}
    # --- review page actions on one sound ---------------------------------------------
    if name in ("toggle", "nudge", "louder", "watch", "remake", "change", "all"):
        st = S.st
        if name == "all":
            for c in st["cues"]:
                if a.get("kind") in (None, "", c["kind"]):
                    c["keep"] = bool(a["keep"])
            E.save(st)
            return {"review": review_view()}
        c = next(x for x in st["cues"] if x["id"] == a["id"])
        if name == "toggle":
            c["keep"] = bool(a.get("keep", not c.get("keep")))
        elif name == "nudge" and c.get("file"):
            c["place_at"] = round(max(0.0, c["place_at"] + float(a["by"])), 3)
        elif name == "louder" and c.get("file"):
            import numpy as np
            import ss_sounds as SS
            db = float(a["db"])
            x = SS.load48(c["file"]) * (10 ** (db / 20))
            peak = float(np.abs(x).max()) if len(x) else 0
            if peak > 0.97:
                x *= 0.97 / peak
            SS.write_wav(x, c["file"])
            c["gain_extra"] = c.get("gain_extra", 0) + db
            c["v"] = c.get("v", 0) + 1                 # makes the window reload the sound
        elif name == "watch":
            return {"video": media_url(E.preview_moment(st, c)) + f"&v={time.time():.0f}"}
        elif name in ("remake", "change"):
            if name == "change":
                import ss_claude as C
                c.update(C.describe(a["words"], c))
            else:
                c["seed_salt"] = str(int(c.get("seed_salt") or 0) + 1)
            c["keep"] = True
            s = choices()
            sec = E.estimate(st, s, n_cues=1)["sounds"]
            return {"job": run_job("Making a new version of that sound",
                                   lambda p, cc, ph: E.make_sounds(st, s, p, cc, ids=[c["id"]]),
                                   "review", [sec])}
        E.save(st)
        return {"cue": cue_view(c)}
    if name == "make_missing":
        st, s = S.st, choices()
        ids = [c["id"] for c in st["cues"] if not c.get("file")]
        sec = E.estimate(st, s, n_cues=len(ids))["sounds"]
        return {"job": run_job("Making the missing sounds",
                               lambda p, c, ph: E.make_sounds(st, s, p, c, ids=ids),
                               "review", [sec])}
    if name == "finish":
        st, s = S.st, choices()
        if not any(c.get("keep") and c.get("file") for c in st["cues"]):
            raise StoryError("No sounds are switched on yet.")
        return {"job": run_job("Putting the sounds in your movie",
                               lambda p, c, ph: E.finish(st, s, p, c), "done",
                               [E.estimate(st, s, n_cues=0)["finish"]])}
    if name == "outputs":
        return {"outputs": outputs_view()}
    if name == "open":
        p = a.get("path")
        if not p or not os.path.exists(p):
            raise StoryError("That file isn't there any more. It may have been moved, renamed "
                             "or still be syncing.")
        os.startfile(p)
        return {"ok": True}
    if name == "gpu":
        return {"gpu": E.gpu_free_gb()}
    if name == "ping":
        return {"ok": True}
    raise StoryError("Unknown request.")


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json", extra=None):
        data = body if isinstance(body, bytes) else body.encode("utf8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        S.last_ping = time.time()
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            html = UI_FILE.read_text(encoding="utf8").replace("__TOKEN__", TOKEN)
            return self._send(200, html, "text/html; charset=utf-8")
        if u.path == "/media":
            if q.get("t", [""])[0] != TOKEN:
                return self._send(403, "{}")
            p = q.get("p", [""])[0]
            if not p or not os.path.isfile(p) or not allowed_media(p):
                return self._send(404, "{}")
            return self._file(p)
        if u.path == "/icon.ico":
            ico = APP_DIR / "storysound.ico"
            return self._send(200, ico.read_bytes() if ico.exists() else b"", "image/x-icon")
        self._send(404, "{}")

    def _file(self, p):
        size = os.path.getsize(p)
        ctype = mimetypes.guess_type(p)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        code = 200
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = min(size - 1, int(m.group(2)))
                code = 206
        length = end - start + 1
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(p, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                    return
                left -= len(chunk)

    def do_POST(self):
        S.last_ping = time.time()
        if self.headers.get("X-Token") != TOKEN:
            return self._send(403, json.dumps({"error": "not allowed"}))
        n = int(self.headers.get("Content-Length") or 0)
        try:
            a = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            a = {}
        name = urlparse(self.path).path.rsplit("/", 1)[-1]
        slow = name in SLOW_ASKS                  # AI questions that take a while: they
        if slow:                                  # don't hold up the rest of the window, and
            S.asking += 1                         # StoryStudio stays open until they finish
        try:
            free = slow or name in NO_LOCK
            with S.lock if not free else _nolock():
                res = api(name, a)
            self._send(200, json.dumps(res))
        except StoryError as ex:
            if slow:
                log("script builder:", name, "said:", ex)
            self._send(200, json.dumps({"error": str(ex)}))
        except Exception:
            log_error("api " + name)
            self._send(200, json.dumps({"error": "Something went wrong. Details were saved "
                                                 "to the log file."}))
        finally:
            if slow:
                S.asking -= 1


class _nolock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# start: server + app window; quit when the window has been closed for a while
# ---------------------------------------------------------------------------
def find_edge():
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if os.path.exists(p):
            return p
    return None


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    url = f"http://127.0.0.1:{port}/"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    edge = find_edge()
    profile = APP_DATA / "window"
    if os.environ.get("STORYSTUDIO_NO_WINDOW"):          # for testing in another browser
        (APP_DATA / "test_url.txt").write_text(url, encoding="utf8")
        while not S.restart:
            time.sleep(2)
        time.sleep(1.5)
        import ss_update
        ss_update.restart()
        srv.shutdown()
        return
    window = None
    if edge:
        window = subprocess.Popen([edge, f"--app={url}", f"--user-data-dir={profile}",
                                   "--window-size=1320,880", "--no-first-run",
                                   "--no-default-browser-check",
                                   "--disable-features=Translate,IntensiveWakeUpThrottling",
                                   "--disable-background-timer-throttling",
                                   "--disable-renderer-backgrounding"],
                                  **core.NOWIN)
    else:
        import webbrowser
        webbrowser.open(url)
    opened = time.time()
    log("StoryStudio window at", url)
    # When does StoryStudio close? When its window is really closed (the Edge window we
    # opened has exited and stopped checking in), or after the window has been silent for a
    # long time with nothing running. Edge slows a background window's check-ins to about
    # once a minute, so a short silence must never close StoryStudio: that left the window
    # open with nothing behind its buttons.
    while True:
        time.sleep(5)
        if S.restart:                          # updated: open the new version, close this one
            time.sleep(1.5)                    # let the window get its answer first
            import ss_update
            ss_update.restart()
            break
        busy = (S.job is not None and not S.job.done) or S.asking > 0
        quiet = time.time() - S.last_ping
        if window is not None and window.poll() is not None and time.time() - opened <= 20:
            window = None        # Edge handed the window to one already open: can't watch it
        closed = window is not None and window.poll() is not None
        if not busy and ((closed and quiet > 30) or quiet > 15 * 60):
            log("StoryStudio closing: window", "closed" if closed else "silent")
            break
    srv.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log_error("server main")
        raise
