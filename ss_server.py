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
              "cartoon": "Cartoon", "crowd": "Crowd", "music": "Music", "other": "Other",
              "shot": "Shot sounds", "voice": "Character sound"}


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
SLOW_ASKS = {"sb_redo", "sb_brainstorm", "sb_cast", "sb_talk"}
PICK_LOCK = threading.Lock()            # one Windows file window at a time
# never wait behind other actions: quick status calls, and file pickers (a Windows file window
# left open must not freeze the rest of StoryStudio)
NO_LOCK = {"job", "ping", "stop", "sb_clips_pick", "guide", "place_folder", "place_sound",
           "place_voices", "client_error", "update_check", "problem_report",
           "sb_pick_text", "sb_pick_song",
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
    if S.st and S.st.get("video"):
        roots.append(Path(S.st["video"]).parent)
    elif S.st:
        proj = Path(S.st["project"]).parent
        roots += [proj / "StoryStudio", proj / "StorySound", proj.parent]
    return any(str(p).lower().startswith(str(r.resolve()).lower()) for r in roots)


def project_view():
    st = S.st
    if not st or st.get("video"):
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
            # ("planned": a run that was stopped while its sounds were being made)
            (st.get("previous", {}).get("cues", 0) > 0 and
             st.get("previous", {}).get("stage") in ("karaoke", "feedback", "planned")),
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
            "choice": S.settings.get("ai", "auto"),
            "claude_model": C.chosen_model(), "chatgpt_model": G.chosen_model(),
            "claude_models": [{"id": i, "name": n, "note": t} for i, n, t in C.CLAUDE_MODELS],
            "model_name": {"claude": C.model_label, "chatgpt": G.model_label}[p]() if p else "",
            "claude_last": ", ".join(C.nice_model(m.strip())
                                     for m in C.last_model().split(",") if m.strip())}


FB_LABEL = {"fix": "Please fix", "suggest": "Idea", "beat": "Beat", "good": "Nicely done"}


def note_view(n):
    return {"id": n["id"], "t": n["time"], "time": fmt_time(n["time"]),
            "end": n.get("end"), "severity": n["severity"],
            "label": FB_LABEL.get(n["severity"], ""), "kind": n.get("kind", ""),
            "title": n["title"], "note": n["note"], "how": n.get("how", ""),
            "keep": bool(n.get("keep", True)), "ai": n.get("source") in ("ai", "claude"),
            "can_do": bool(n.get("edit"))}


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
            "can_retime": fb["kind"] == "cartoon" and any(n.get("edit") for n in fb["notes"]),
            "retimed": out.get("retimed", ""),
            "snapshot_time": st.get("snapshot_time", "")}


def cue_view(c):
    return {"id": c["id"], "time": fmt_time(c.get("place_at", c["start"])),
            "end": fmt_time(c["end"]), "what": c["what"],
            "kind": KIND_NAMES.get(c["kind"], c["kind"]), "kind_id": c["kind"],
            "keep": bool(c.get("keep")), "made": bool(c.get("file")),
            "source": {"pc": "Made on this PC", "fal": "Made with Fal", "freesound": "Freesound",
                       "adobe": "Not made"}.get(c.get("source", ""), ""),
            "audio": media_url(c.get("file")), "prompt": c.get("sound_prompt", ""),
            "adobe": c.get("adobe_prompt", ""), "suno": c.get("suno_prompt", ""),
            "note": c.get("note", ""), "credit": c.get("credit", ""),
            "volume": c.get("gain_extra", 0), "added": bool(c.get("added_by_check")),
            # the judge wasn't sure this one sounds like its description: worth a listen
            "unsure": c.get("kind") != "shot" and bool(c.get("file")) and
            c.get("match") is not None and c["match"] < 0.16,
            "frame": round(1.0 / float((S.st or {}).get("fps") or 24.0), 4)}


def review_view():
    st = S.st
    if not st:
        return None
    return {"title": st["seq_name"], "notes": st.get("plan_notes", ""),
            "notice": st.get("notice", ""), "levels": st.get("levels", []),
            "can_voice": bool(st.get("cartoon") or st.get("movie_type") == "cartoon"),
            "voices": sum(1 for c in st.get("cues", []) if c["kind"] == "voice"),
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
            "source": (st or {}).get("project", ""), "seq_uid": (st or {}).get("seq_uid", ""),
            "video_out": out.get("video_out", ""), "video_source": (st or {}).get("video", "")}


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
            if str(ex) != "Stopped.":                # keep a note of what the user was told
                log("job stopped with a message:", title, "->", ex)
        except Exception:
            log_error("job " + title)
            job.failed = True
            job.error = ("Something went wrong. Details were saved to the log file in the "
                         "StoryStudio folder.")
        finally:
            job.done = True
            try:
                import storysound as E
                if S.st and S.st.get("stage") != "opened":   # nothing to keep from a
                    E.save(S.st)                             # project that was only opened
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


def native_folder(title, initial=None):
    """A normal Windows "choose a folder" window."""
    import tkinter as tk
    from tkinter import filedialog
    if not PICK_LOCK.acquire(blocking=False):
        raise StoryError("A file window is already open. Choose there (it may be behind this "
                         "window), or close it, then try again.")
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.lift()
        root.focus_force()
        try:
            return filedialog.askdirectory(title=title, mustexist=True,
                                           initialdir=initial or core.documents_dir())
        finally:
            root.destroy()
    finally:
        PICK_LOCK.release()


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
                    "busy", "add_music", "use_local", "use_freesound", "best_of",
                    "sound_source", "fal_limit", "karaoke_rows", "karaoke_style", "karaoke_color",
                    "karaoke_highlight")},
                "has_fs_key": bool(S.settings.get("freesound_key")),
                "has_hf_key": bool(S.settings.get("hf_key")),
                "has_fal_key": bool(S.settings.get("fal_key")),
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
        v["chatgpt_models"] = G.models()         # asked from the ChatGPT program: always current
        v["model_name"] = ai_view()["model_name"]    # (now that the list has the names)
        return {"ai": v}
    if name == "ai_model":
        which = "chatgpt_model" if a.get("which") == "chatgpt" else "claude_model"
        m = str(a.get("model") or "").strip()
        if m and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:\[\]-]{0,79}", m):
            raise StoryError("That doesn't look like a model's name. Use the name exactly as "
                             "Claude or ChatGPT writes it, for example claude-opus-5-5.")
        S.settings[which] = m
        core.save_settings(S.settings)
        return {"ai": ai_view()}
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
    if name == "pick_video":
        import ss_karaoke as K
        f = native_pick("Choose your video", [("Videos", K.VIDEO_FILES), ("All", "*.*")],
                        S.settings.get("last_video_folder") or S.settings.get("last_folder"))
        if not f:
            return {"video": None}
        a = {"path": f}
        name = "open_video"
    if name == "open_video":
        import ss_karaoke as K
        st = K.open_video(a["path"])
        S.settings["last_video_folder"] = str(Path(a["path"]).parent)
        core.save_settings(S.settings)
        S.st = st
        return {"video": {"name": Path(st["video"]).name, "path": st["video"],
                          "duration": fmt_time(st["duration"]),
                          "size": f"{st['width']}×{st['height']}",
                          "thumb": media_url(st.get("thumb"))}}
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
        return {"text": E.nice_minutes(total), "sounds": e["n"], "busy_gpu": e["busy_gpu"],
                "shots": e.get("shots"), "limit": e.get("limit"),
                "full_cost": round(e["full_cost"], 2) if e.get("full_cost") else None,
                "share": e.get("share"),
                "cost": round(e["cost"], 2) if "cost" in e else None}
    if name == "save_settings":
        for k in ("busy", "add_music", "use_local", "use_freesound", "best_of", "sound_source"):
            if k in a:
                S.settings[k] = a[k]
        if "fal_limit" in a:
            try:
                S.settings["fal_limit"] = max(0.0, round(float(a["fal_limit"] or 0), 2))
            except (TypeError, ValueError):
                S.settings["fal_limit"] = 0.0
        for k in ("freesound_key", "hf_key", "fal_key"):
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
    if name == "karaoke_sample":
        import base64
        import ss_karaoke as K
        st = S.st or {}
        s = dict(S.settings, **{k: a[k] for k in ("karaoke_rows", "karaoke_style",
                                                  "karaoke_color", "karaoke_highlight") if k in a})
        out = core.APP_DATA / "karaoke_sample.jpg"
        K.sample(st, s, out)
        return {"image": "data:image/jpeg;base64," + base64.b64encode(out.read_bytes()).decode()}
    if name == "start_karaoke":
        import ss_karaoke as K
        st, s = S.st, choices()
        text = a.get("lyrics", "")
        if not K.parse_lyrics(text):
            raise StoryError("Please paste the song's lyrics first.")
        st["lyrics_text"] = text
        for k in ("karaoke_rows", "karaoke_style", "karaoke_color",
                  "karaoke_highlight"):                       # remembered for next time
            if a.get(k) is not None:
                S.settings[k] = s[k] = a[k]
        core.save_settings(S.settings)
        est = 40 + st["duration"] * 0.6 + (0 if st.get("picture") else 30)
        return {"job": run_job("Making karaoke lyrics",
                               lambda p, c, ph: K.karaoke(st, s, text, p, c), "done", [est])}
    # --- script builder (no Premiere project needed) ---------------------------------
    if name == "guide":
        import ss_craft
        return {"lessons": [{"title": t, "text": x} for t, x in ss_craft.LESSONS]}
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
                                       a.get("style_name", ""),
                                       onscreen=a.get("onscreen") or "auto")}
    if name == "sb_cast":
        import ss_script as SB
        return {"characters": SB.suggest_cast(a)}
    if name == "sb_talk":
        import ss_script as SB
        return SB.talk(a.get("spec") or {}, a.get("rounds") or [])
    if name == "sb_start":
        import math
        import ss_script as SB
        spec = dict(a)
        if spec.get("kind") == "story":
            secs = len((spec.get("text") or "").split()) / SB.WORDS_PER_SEC
            clip = 8
        elif spec.get("kind") == "cartoon":
            secs, clip = float(spec.get("cartoon_seconds") or 90), int(spec.get("clip") or 5)
        else:
            secs, clip = float(spec.get("song_seconds") or 0), int(spec.get("clip") or 8)
        shots = secs / clip * (1 + SB.EXTRA)
        est = 70 + 65 * math.ceil(max(1, shots) / 24)

        def work(progress, cancel, phase):
            S.sb = SB.build(spec, progress, cancel)
        return {"job": run_job("Planning your shots", work, "script", [est])}
    if name == "sb_result":
        import ss_script as SB
        return {"result": S.sb, "shot_types": SB.type_menu()}
    if name == "sb_clips_pick":
        if not S.sb:
            raise StoryError("The shot list isn't open any more. Open it again, then try once "
                             "more.")
        f = native_folder("Choose the folder with your pictures and clips",
                          S.sb.get("clip_folder") or S.settings.get("last_clip_folder"))
        if f:
            S.settings["last_clip_folder"] = str(Path(f).parent)
            core.save_settings(S.settings)
        return {"folder": f or ""}
    if name == "sb_clips_check":
        import ss_clips as K
        if not S.sb:
            raise StoryError("The shot list isn't open any more. Open it again, then try once "
                             "more.")
        r, folder = S.sb, a.get("folder") or S.sb.get("clip_folder") or ""
        found, _, _ = K.find_files(r, folder)
        n = sum(len(v) for v in found.values())
        if not n:
            raise StoryError("StoryStudio didn't find any pictures or clips with a shot number "
                             "in their name in that folder. Name them like “shot 5.png” and "
                             "“shot 5.mp4”.")

        def work(progress, cancel, phase):
            K.check(r, folder, progress, cancel, again=bool(a.get("again")))
        return {"job": run_job("Looking at your pictures and clips", work, "script",
                               [20 + n * 1.2 + (n // 15 + 1) * 45])}
    if name == "place_folder":
        import ss_place as PL
        f = native_folder("Choose the folder with your clips",
                          S.settings.get("last_clip_folder"))
        if not f:
            return {"folder": ""}
        S.settings["last_clip_folder"] = str(Path(f).parent)
        core.save_settings(S.settings)
        return PL.folder_view(f)
    if name == "place_sound":
        f = native_pick("Choose the narration or the song",
                        [("Sound", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.wma *.aif *.aiff"),
                         ("All", "*.*")], a.get("folder") or S.settings.get("last_folder"))
        return {"path": f or "", "name": Path(f).name if f else ""}
    if name == "place_voices":
        import ss_place as PL
        f = native_folder("Choose the folder with the voice files", a.get("folder") or
                          S.settings.get("last_clip_folder"))
        if not f:
            return {"path": ""}
        v = PL.voices_view(f)
        if not v["count"]:
            raise StoryError("There are no sound files in that folder.")
        return v
    if name == "place_start":
        import ss_place as PL
        spec = {k: a.get(k) or "" for k in ("folder", "script", "audio", "voices", "kind",
                                            "title")}
        found = PL.look(spec["folder"])[0] if spec["folder"] else []
        if not found:
            raise StoryError("StoryStudio didn't find any clips with a shot number in their "
                             "name in that folder. Name them like “Shot 5.mp4”.")
        n = len(found)
        sound = 0.0
        if spec["voices"]:
            sound = 60.0 * PL.voices_view(spec["voices"])["count"]      # a guess: long enough
        elif spec["audio"]:
            try:
                sound = float(core.probe(spec["audio"]).get("duration") or 0)
            except Exception:
                sound = 0.0

        def work(progress, cancel, phase):
            S.sb = PL.build(spec, progress, cancel)
        return {"job": run_job("Placing your clips", work, "script",
                               [15 + n * 0.6 + (90 + n * 2 if spec["script"] else 0)
                                + sound * 0.12])}
    if name == "sb_roughcut":
        import ss_roughcut as RC
        if not S.sb:
            raise StoryError("The shot list isn't open any more. Open it again, then try once "
                             "more.")
        res = RC.build(S.sb, a.get("folder") or S.sb.get("clip_folder") or "")
        return {"rough": res, "result": S.sb}
    if name == "reveal":
        p = a.get("path")
        if not p or not os.path.exists(p):
            raise StoryError("That file isn't there any more. It may have been moved, renamed "
                             "or still be syncing.")
        subprocess.Popen(["explorer", "/select,", os.path.normpath(p)])
        return {"ok": True}
    if name == "sb_shot_type":
        import ss_script as SB
        if not S.sb:
            raise StoryError("The shot list isn't open any more. Open it again from the "
                             "script builder, then try once more.")
        return SB.set_type(S.sb, a.get("n", 0), a.get("code", ""))
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
        return {"result": S.sb, "shot_types": SB.type_menu()}
    if name == "update_check":
        import ss_update
        info = ss_update.check(timeout=8)
        return {"update": {"version": info["version"], "notes": info.get("notes", "")}
                if info else None, "current": ss_update.local_version()}
    if name == "update_now":
        import ss_update
        if (S.job and not S.job.done) or S.asking > 0:
            raise StoryError("StoryStudio is busy. Update when it has finished.")
        info = ss_update.check(timeout=10)
        if not info:
            return {"ok": False, "message": "You already have the newest version."}
        if not ss_update.apply(info):
            raise StoryError("The update couldn't be downloaded. Check the internet and try "
                             "again; your StoryStudio still works as it is.")
        log("updated to", info["version"], "- restarting")
        try:                                  # so the new version reopens what was open
            st = S.st if S.st and not S.st.get("video") else {}
            (core.APP_DATA / "resume.json").write_text(json.dumps({
                "page": str(a.get("page") or ""), "at": time.time(),
                "project": st.get("project", ""), "seq_uid": st.get("seq_uid", ""),
                "shot_list": ((S.sb or {}).get("files") or {}).get("json", "")}),
                encoding="utf8")
        except Exception:
            log_error("resume note")
        S.restart = True
        return {"ok": True, "version": info["version"]}
    if name == "resume":
        # after "Update now": go back to the shot list, the sounds or the project that was
        # open (the note is used once, and only if it is fresh)
        f = core.APP_DATA / "resume.json"
        try:
            note = json.loads(f.read_text(encoding="utf8"))
            f.unlink()
        except (OSError, ValueError):
            return {"view": ""}
        if time.time() - float(note.get("at") or 0) > 900:
            return {"view": ""}
        page, proj, sl = note.get("page"), note.get("project") or "", note.get("shot_list") or ""
        try:
            if page == "script" and sl and os.path.exists(sl):
                import ss_script as SB
                S.sb = SB.load(sl)
                return {"view": "script", "result": S.sb, "shot_types": SB.type_menu()}
            if page in ("project", "review", "done", "feedback") and proj and os.path.exists(proj):
                S.st = E.open_project(proj, note.get("seq_uid") or None)
                S.st["thumb"] = make_thumb(S.st)
                view = project_view()
                if page in ("review", "done") and view.get("previous"):
                    prev = E.load(S.st["work"], "cues")
                    prev["story"] = S.st.get("story") or prev.get("story")
                    prev["thumb"] = S.st.get("thumb")
                    S.st = prev
                    return {"view": "review", "project": view, "review": review_view()}
                return {"view": "project", "project": view}
        except StoryError as e:
            log("resume skipped:", e)
        except Exception:
            log_error("resume")
        return {"view": ""}
    if name == "client_error":
        log("window problem:", str(a.get("msg", ""))[:3000])
        return {"ok": True}
    if name == "start_feedback":
        import math
        import ss_feedback as F
        st, s = S.st, choices()
        kind = a.get("kind") or "auto"
        focus = "timing" if a.get("focus") == "timing" else "all"
        use_claude = bool(a.get("claude", True)) or focus == "timing"
        dur = st["duration"]
        est = 25 + dur * 0.12 + (0 if st.get("picture") or st.get("export") else dur * 0.35)
        n_shots = (st.get("info") or {}).get("shots") or max(4, round(dur / 5))
        if use_claude and focus != "timing":
            est += 20 + 75 * math.ceil(dur / (160 if dur <= 420 else 320))
        if use_claude:                              # the shot-by-shot timing look
            est += 20 + n_shots * 0.6 + math.ceil(math.ceil(n_shots / 6) / 3) * 45 + 30
        st["feedback_dirty"] = False
        return {"job": run_job("Checking your timing and pacing" if focus == "timing"
                               else "Watching your edit",
                               lambda p, c, ph: F.feedback(st, s, p, c, kind, use_claude, focus),
                               "feedback", [est])}
    if name == "feedback":
        return {"feedback": feedback_view()}
    if name == "open_previous_feedback":
        prev = E.load(S.st["work"], "feedback")
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
    if name == "fb_retime":
        import ss_feedback as F
        if (S.st.get("feedback") or {}).get("kind") != "cartoon":
            raise StoryError("StoryStudio only makes timing changes itself on silent cartoons, "
                             "where nothing has to stay in step with a song or narration.")
        res = F.retime(S.st)
        return {"project": res["project"], "changed": res["changed"],
                "shorter": res["shorter"], "skipped": res["skipped"],
                "feedback": feedback_view()}
    if name == "fb_save":
        import ss_feedback as F
        F.write_outputs(S.st)
        S.st["feedback_dirty"] = False
        E.save(S.st)
        return {"feedback": feedback_view()}
    if name == "open_previous":
        prev = E.load(S.st["work"], "cues")
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
    if name == "add_voices":
        st, s = S.st, choices()
        n = len(st.get("shots") or [])
        return {"job": run_job("Adding the characters' own sounds",
                               lambda p, c, ph: E.add_voices(st, s, p, c), "review",
                               [150 + 9 * n])}
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
    if name == "problem_report":
        path = core.problem_report()
        try:                                     # show it, selected, in its folder
            subprocess.Popen(["explorer", "/select,", path])
        except OSError:
            pass
        return {"path": path, "name": Path(path).name,
                "where": "your Desktop" if Path(path).parent != core.APP_DATA
                else "the StoryStudio folder"}
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
        try:
            self.end_headers()
            self.wfile.write(data)
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            pass                                   # the window went away (closed or reloaded)

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
            log("message shown:", name, "->", ex)
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


def window_open(profile):
    """Is StoryStudio's window still open? It runs in Edge with StoryStudio's own profile, so
    any Edge process using that profile means yes (even when Edge has frozen the hidden
    window and it can't check in). None if this can't be told."""
    key = f"--user-data-dir={profile}".lower()
    try:
        import psutil
        for p in psutil.process_iter(["name"]):
            if (p.info.get("name") or "").lower() != "msedge.exe":
                continue
            try:
                if any(key in a.lower() for a in p.cmdline()):
                    return True
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        return False
    except ImportError:
        pass
    except Exception:
        log_error("window check")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "Get-CimInstance Win32_Process -Filter \"Name='msedge.exe'\" | "
                            "ForEach-Object { $_.CommandLine }"], capture_output=True,
                           text=True, timeout=30, **core.NOWIN)
        if r.returncode == 0:
            return key in (r.stdout or "").lower()
    except Exception:
        pass
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
    # When does StoryStudio close? When its window is really closed: the Edge window we
    # opened has exited, or (when Edge handed the window to one already running, or the
    # window has gone quiet) no Edge process is using StoryStudio's window profile any more.
    # Silence alone is not enough: Edge slows or freezes a hidden window (while working in
    # Premiere) for a long time, and closing then left the window open with nothing behind
    # its buttons. Only after many hours of silence with no way to tell does it give up.
    checked = 0.0
    tick = time.time()
    while True:
        time.sleep(5)
        if time.time() - tick > 60:            # the PC was asleep: the window hasn't had a
            S.last_ping = time.time()          # chance to check in yet, so start counting again
        tick = time.time()
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
        if busy:
            continue
        if closed and quiet > 30:
            log("StoryStudio closing: window closed")
            break
        if quiet > 90 and time.time() - checked > 60:
            checked = time.time()
            open_now = window_open(profile) if edge else None
            if open_now is False:
                log("StoryStudio closing: window closed (no StoryStudio window in Edge)")
                break
            if open_now is None and quiet > 8 * 3600:
                log("StoryStudio closing: window silent for hours")
                break
    srv.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log_error("server main")
        raise
