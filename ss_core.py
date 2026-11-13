"""StoryStudio shared helpers: app folders, ffmpeg, settings, story reading, logging."""
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path


APP_NAME = "StoryStudio"
APP_DIR = Path(__file__).resolve().parent


def _home():
    """StoryStudio's own folder (app, venv, models, work). Not under AppData: the Claude
    desktop app and other packaged apps redirect AppData writes into a private copy that
    Windows shortcuts can't see, so StoryStudio lives in C:\\Users\\<name>\\StoryStudio."""
    for k in ("STORYSTUDIO_HOME", "STORYSOUND_HOME"):
        if os.environ.get(k):
            return Path(os.environ[k])
    if APP_DIR.name.lower() == "app" and (APP_DIR.parent / "venv").exists():
        return APP_DIR.parent
    return Path.home() / APP_NAME


APP_DATA = _home()
MODELS = APP_DATA / "models"
NOWIN = {"creationflags": 0x08000000} if os.name == "nt" else {}
TICKS = 254016000000          # Premiere ticks per second

APP_DATA.mkdir(parents=True, exist_ok=True)


class StoryError(Exception):
    """An error with a plain message that is safe to show in the window."""


# ---------------------------------------------------------------------------
# logging and settings
# ---------------------------------------------------------------------------
def log(*parts):
    try:
        with open(APP_DATA / "storysound_log.txt", "a", encoding="utf8") as f:
            f.write(" ".join(str(p) for p in parts) + "\n")
    except OSError:
        pass


def log_error(where):
    log("ERROR in", where, "\n" + traceback.format_exc())


SETTINGS_FILE = APP_DATA / "settings.json"
DEFAULTS = {
    "use_local": True,
    "use_freesound": False,
    "freesound_key": "",
    "busy": "normal",          # light / normal / lots
    "add_music": False,        # the user makes music in Suno; StoryStudio only on request
    "best_of": 3,              # takes per sound; a judge AI keeps the best one
    "hf_key": "",              # Hugging Face key: Stable Audio Open for backgrounds
    "style": "auto",           # auto / cartoon / realistic
    "loudness_target": -14.0,  # YouTube
    "last_folder": "",
}


def load_settings():
    s = dict(DEFAULTS)
    try:
        s.update(json.loads(SETTINGS_FILE.read_text(encoding="utf8")))
    except (OSError, ValueError):
        pass
    return s


def save_settings(s):
    try:
        SETTINGS_FILE.write_text(json.dumps(s, indent=2), encoding="utf8")
    except OSError:
        log_error("save_settings")


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------
def _find_tool(name):
    exe = name + (".exe" if os.name == "nt" else "")
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    for p in (APP_DATA / "bin" / exe, APP_DIR / "bin" / exe,
              local / "CleanView" / "app" / "bin" / exe,
              local / "CleanListen" / "app" / "bin" / exe):
        if p.exists():
            return str(p)
    return shutil.which(name)


def ffmpeg():
    p = _find_tool("ffmpeg")
    if not p:
        raise StoryError("StoryStudio still needs ffmpeg (the free tool that reads video). "
                         "Go to the home screen and click \"Download now\".")
    return p


FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"


def download_ffmpeg(progress=None):
    """StoryStudio's own copy of ffmpeg and ffprobe in APP_DATA/bin (no other app needed)."""
    import urllib.request
    import zipfile
    bin_dir = APP_DATA / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    tmp = bin_dir / "ffmpeg-download.zip"
    req = urllib.request.Request(FFMPEG_URL, headers={"User-Agent": "StoryStudio"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            got += len(b)
            if progress and total:
                progress(got / total)
    with zipfile.ZipFile(tmp) as z:
        for name in z.namelist():
            base = name.rsplit("/", 1)[-1].lower()
            if base in ("ffmpeg.exe", "ffprobe.exe"):
                with z.open(name) as src, open(bin_dir / base, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    tmp.unlink()
    if not (bin_dir / "ffmpeg.exe").exists():
        raise StoryError("StoryStudio couldn't set up ffmpeg. Check the internet and try again.")
    return str(bin_dir / "ffmpeg.exe")


def ffprobe():
    p = _find_tool("ffprobe")
    if not p:
        raise StoryError("StoryStudio can't find ffprobe. Please run the installer again.")
    return p


def run(cmd, timeout=None, check=True):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf8",
                       errors="replace", timeout=timeout, **NOWIN)
    if check and r.returncode != 0:
        log("command failed:", cmd, "\n", (r.stderr or "")[-2000:])
        raise StoryError("A video step didn't work. Details were saved to the log.")
    return r


def probe(path):
    """Duration, frame rate, size and audio info of a media file."""
    r = run([ffprobe(), "-v", "error", "-show_entries",
             "format=duration:stream=codec_type,width,height,r_frame_rate,channels,sample_rate",
             "-of", "json", path])
    d = json.loads(r.stdout or "{}")
    info = {"duration": float(d.get("format", {}).get("duration") or 0), "fps": 0.0,
            "width": 0, "height": 0, "has_video": False, "has_audio": False, "channels": 0}
    for s in d.get("streams", []):
        if s.get("codec_type") == "video" and not info["has_video"]:
            num, _, den = (s.get("r_frame_rate") or "0/1").partition("/")
            try:
                info["fps"] = float(num) / float(den or 1)
            except (ValueError, ZeroDivisionError):
                info["fps"] = 0.0
            info.update(width=s.get("width", 0), height=s.get("height", 0), has_video=True)
        elif s.get("codec_type") == "audio" and not info["has_audio"]:
            info.update(has_audio=True, channels=int(s.get("channels") or 2),
                        sample_rate=int(s.get("sample_rate") or 48000))
    return info


def extract_audio(src, out, rate=48000, channels=2, start=None, length=None):
    cmd = [ffmpeg(), "-y", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", src]
    if length is not None:
        cmd += ["-t", f"{length:.3f}"]
    cmd += ["-vn", "-ac", str(channels), "-ar", str(rate), "-c:a", "pcm_s16le", out]
    run(cmd)
    return out


# ---------------------------------------------------------------------------
# story text
# ---------------------------------------------------------------------------
def read_story(path):
    """Plain text from a .docx, .txt, .md, .rtf or .pdf story/script."""
    p = Path(path)
    ext = p.suffix.lower()
    try:
        if ext == ".docx":
            with zipfile.ZipFile(p) as z:
                xml = z.read("word/document.xml").decode("utf8", "replace")
            xml = re.sub(r"</w:p>", "\n", xml)
            xml = re.sub(r"<w:tab/>", "\t", xml)
            xml = re.sub(r"<w:br/>", "\n", xml)
            text = re.sub(r"<[^>]+>", "", xml)
            import html
            text = html.unescape(text)
        elif ext == ".pdf":
            text = _pdf_text(p)
        elif ext == ".rtf":
            raw = p.read_text(encoding="utf8", errors="replace")
            text = re.sub(r"\\[a-z]+-?\d* ?|[{}]", "", raw)
        else:
            text = p.read_text(encoding="utf8", errors="replace")
    except (OSError, zipfile.BadZipFile, KeyError):
        log_error("read_story")
        raise StoryError("StoryStudio couldn't read that story file. Try saving it as a Word "
                         "document (.docx) or a text file (.txt).")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _pdf_text(p):
    try:
        import pypdf
    except ImportError:
        try:                                  # one-time install (updates don't add packages)
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "pypdf"],
                           check=True, timeout=300, capture_output=True, **NOWIN)
            import pypdf
        except Exception:
            log_error("install pypdf")
            raise StoryError("StoryStudio couldn't read that PDF. Save it as a Word document "
                             "or text file instead.")
    text = "\n".join((pg.extract_text() or "") for pg in pypdf.PdfReader(str(p)).pages)
    if text.count("\n") > len(text) / 12:     # some PDFs come out one word per line
        text = re.sub(r"[^\S\n]*\n[^\S\n]*", " ", text)
        text = re.sub(r"[^\S\n]{2,}", " ", text)
    return text


STORY_EXTS = (".docx", ".txt", ".md", ".rtf", ".pdf")


def find_story(near, title=""):
    """Look for a story/script file near a project or movie. Returns a path or ''."""
    near = Path(near)
    folders = []
    for f in (near if near.is_dir() else near.parent, ):
        folders += [f, f.parent, f.parent.parent]
    best, best_score = "", 0
    words = [w for w in re.findall(r"[a-z]+", title.lower()) if len(w) > 2]
    seen = set()
    for folder in folders:
        if not folder.exists() or folder in seen:
            continue
        seen.add(folder)
        try:
            files = list(folder.glob("*"))
        except OSError:
            continue
        for f in files:
            if f.suffix.lower() not in STORY_EXTS or f.name.startswith("~$"):
                continue
            n = f.stem.lower()
            score = 1
            if any(k in n for k in ("story", "script", "narration", "screenplay")):
                score += 3
            score += sum(1 for w in words if w in n)
            if score > best_score:
                best, best_score = str(f), score
    return best if best_score >= 2 else ""


def documents_dir():
    """The Documents folder the user really sees (often OneDrive\\Documents, not the old
    local C:\\Users\\<name>\\Documents)."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            import uuid as _uuid
            fid = _uuid.UUID("{FDD39AD0-238F-46AF-ADB4-6C85480369C7}")   # FOLDERID_Documents
            buf = ctypes.c_wchar_p()
            guid = (ctypes.c_byte * 16).from_buffer_copy(fid.bytes_le)
            if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None,
                                                           ctypes.byref(buf)) == 0:
                p = buf.value
                ctypes.windll.ole32.CoTaskMemFree(buf)
                if p and os.path.isdir(p):
                    return p
        except Exception:
            pass
    return str(Path.home() / "Documents")


def _known_folder(folder_id, fallback):
    if os.name == "nt":
        try:
            import ctypes
            import uuid as _uuid
            fid = _uuid.UUID(folder_id)
            buf = ctypes.c_wchar_p()
            guid = (ctypes.c_byte * 16).from_buffer_copy(fid.bytes_le)
            if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None,
                                                           ctypes.byref(buf)) == 0:
                p = buf.value
                ctypes.windll.ole32.CoTaskMemFree(buf)
                if p and os.path.isdir(p):
                    return p
        except Exception:
            pass
    return str(Path.home() / fallback)


def desktop_dir():
    """The Desktop the user really sees (often OneDrive's)."""
    return _known_folder("{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}", "Desktop")


def _tail(path, lines, skip=()):
    try:
        text = Path(path).read_text(encoding="utf8", errors="replace").splitlines()
    except OSError:
        return []
    return [x for x in text if not any(s in x for s in skip)][-lines:]


def problem_report():
    """One text file on the Desktop with everything needed to work out what went wrong on
    this computer: the version, what the computer has, the settings (no keys) and the last
    part of StoryStudio's own notes. Returns its path."""
    import platform
    import time
    out = ["StoryStudio problem report", "=" * 60, time.strftime("Made %Y-%m-%d %H:%M"), ""]

    def add(label, fn):
        try:
            out.append(f"{label}: {fn()}")
        except Exception as ex:
            out.append(f"{label}: (couldn't tell: {type(ex).__name__})")

    add("StoryStudio version", lambda: json.loads(
        (APP_DIR / "version.json").read_text(encoding="utf8")).get("version"))
    add("Windows", platform.platform)
    add("Python", lambda: sys.version.split()[0])
    add("Processor cores", os.cpu_count)

    def memory():
        import ctypes

        class MEM(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + \
                       [(n, ctypes.c_ulonglong) for n in ("total", "avail", "tp", "ap", "tv",
                                                          "av", "aev")]
        m = MEM()
        m.length = ctypes.sizeof(MEM)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return f"{m.total / 2 ** 30:.0f} GB, {m.avail / 2 ** 30:.0f} GB free"
    add("Memory", memory)

    def card():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                                "--format=csv,noheader"], capture_output=True, text=True,
                               timeout=8, **NOWIN)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip().replace("\n", " | ")
        except (OSError, subprocess.SubprocessError):
            pass
        return ("no NVIDIA graphics card found (sounds are made on the processor, which is "
                "much slower)")
    add("Graphics card", card)
    add("Free disk space", lambda: f"{shutil.disk_usage(APP_DATA).free / 2 ** 30:.0f} GB "
                                   f"on the drive with {APP_DATA}")

    def tools():
        import importlib.util
        have = [n for n in ("torch", "mmaudio", "demucs", "sherpa_onnx", "soundfile", "scipy")
                if importlib.util.find_spec(n) is not None]
        return ", ".join(have) or "none"
    add("AI parts installed", tools)
    add("ffmpeg", lambda: "found" if Path(ffmpeg()).exists() else "MISSING")

    def helper():
        import ss_claude as C
        import ss_gpt as G
        return (f"Claude {'on this computer' if C.find_claude() else 'not installed'}; "
                f"ChatGPT {'set up' if G.find() else 'not set up'}; using {C.provider()}")
    add("AI helper", helper)

    def settings():
        s = load_settings()
        return {k: (("(set)" if v else "(not set)") if "key" in k.lower() else v)
                for k, v in s.items() if k not in ("recent", "last_folder")}
    add("Settings", settings)

    def section(title, lines):
        if lines:
            out.extend(["", title, "-" * 60] + lines)
    section("The last things StoryStudio noted (newest at the bottom)",
            _tail(APP_DATA / "storysound_log.txt", 400))
    logs = ("generated/pc_worker_log.txt", "generated/pick.log", "generated/sao_jobs.log",
            "worker_log.txt")
    try:                                         # the helper programs' own notes: the two
        runs = sorted((p for p in (APP_DATA / "work").iterdir() if p.is_dir()),   # newest runs
                      key=lambda p: -max([(p / r).stat().st_mtime for r in logs
                                          if (p / r).exists()] or [0]))
        runs = [p for p in runs if any((p / r).exists() for r in logs)][:2]
    except OSError:
        runs = []
    for run_dir in runs:
        for rel in logs:
            section(f"{run_dir.name}: {rel}",
                    _tail(run_dir / rel, 40, skip=("too short", "Truncating", "HF_TOKEN",
                                                   "iB/s]")))
    section("Installing", _tail(APP_DATA / "install_log.txt", 30))
    name = time.strftime("StoryStudio problem report %Y-%m-%d %H%M.txt")
    path = Path(desktop_dir()) / name
    try:
        path.write_text("\n".join(str(x) for x in out), encoding="utf8")
    except OSError:                              # no Desktop to write to: the StoryStudio folder
        path = APP_DATA / name
        path.write_text("\n".join(str(x) for x in out), encoding="utf8")
    return str(path)


def fmt_time(sec):
    sec = max(0.0, float(sec))
    m, s = divmod(sec, 60)
    return f"{int(m)}:{s:04.1f}"


def safe_name(s):
    return re.sub(r'[<>:"/\\|?*]+', "-", s).strip() or "movie"
