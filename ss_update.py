"""StoryStudio's auto-updater.

New versions are published to GitHub (BookWorm824/storystudio, branch main). version.json there
lists the version number and every file. When it is higher than this copy's version, every file
is downloaded into a holding folder, Python files are compiled and data files parsed so a
broken download is never installed, and only then are they copied over this copy.

Rules for releasing (same as CleanView):
- bump "version" in version.json every release;
- list every app file in "files" (flat names, no folders);
- new pip packages are not installed by the updater: install them at runtime when first
  needed (see ss_core._pdf_text) and add them to installer/install.ps1.
"""
import json
import os
import py_compile
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parent
RAW = os.environ.get("STORYSTUDIO_UPDATE_URL",
                     "https://raw.githubusercontent.com/BookWorm824/storystudio/main/")
REPO = "BookWorm824/storystudio"
NOWIN = {"creationflags": 0x08000000} if os.name == "nt" else {}


def _vt(v):
    try:
        return tuple(int(x) for x in str(v).strip().split("."))
    except ValueError:
        return (0,)


def local_version():
    try:
        return json.loads((APP / "version.json").read_text(encoding="utf8"))["version"]
    except (OSError, ValueError, KeyError):
        return "0"


def _get(name, timeout=20, base=None):
    req = urllib.request.Request((base or RAW) + name,
                                 headers={"User-Agent": "StoryStudio-updater",
                                          "Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _pinned_base():
    """The address of one exact snapshot of the published files. GitHub keeps each file of
    the "main" address in its own short-lived cache, so right after a release some files can
    still be the old ones; a snapshot address always gives files that belong together.
    None when it can't be found (the "main" address is used then, as before)."""
    if "STORYSTUDIO_UPDATE_URL" in os.environ:
        return None
    try:
        req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/commits/main",
                                     headers={"User-Agent": "StoryStudio-updater",
                                              "Accept": "application/vnd.github.sha"})
        with urllib.request.urlopen(req, timeout=10) as r:
            sha = r.read().decode("ascii").strip()
        if len(sha) == 40 and all(c in "0123456789abcdef" for c in sha):
            return f"https://raw.githubusercontent.com/{REPO}/{sha}/"
    except Exception:
        pass
    return None


def check(timeout=8):
    """The newer version's info ({"version", "files", "notes"}) or None."""
    try:
        info = json.loads(_get("version.json", timeout).decode("utf8"))
    except Exception:
        return None
    if _vt(info.get("version")) > _vt(local_version()):
        return info
    return None


def apply(info, progress=None):
    """Download and install the new version. Returns True when installed."""
    base = _pinned_base()
    if base:
        try:                                 # the snapshot's own list of files and version
            pinned = json.loads(_get("version.json", 20, base).decode("utf8"))
            if _vt(pinned.get("version")) >= _vt(info.get("version")):
                info = pinned
            else:
                base = None
        except Exception:
            base = None
    files = [f for f in info.get("files", []) if f and "/" not in f and "\\" not in f
             and not f.startswith(".")]
    if "version.json" not in files:
        files.append("version.json")
    hold = APP / "_update"
    shutil.rmtree(hold, ignore_errors=True)
    hold.mkdir()
    try:
        for i, f in enumerate(files):
            if progress:
                progress(i / len(files), f"Downloading {f}")
            data = _get(f, timeout=60, base=base)
            (hold / f).write_bytes(data)
            if f.endswith((".py", ".pyw")):
                py_compile.compile(str(hold / f), doraise=True)
            elif f.endswith(".json"):
                json.loads(data.decode("utf8"))
            elif f.endswith(".html") and b"</html>" not in data[-200:]:
                raise ValueError(f"{f} looks incomplete")
        for f in files:                      # everything checked: now swap them in
            if f != "version.json":
                shutil.copy2(hold / f, APP / f)
        shutil.copy2(hold / "version.json", APP / "version.json")   # last, so a failed
        return True                                                  # swap retries next time
    except Exception as ex:
        _log(f"update to {info.get('version')} failed: {type(ex).__name__}: {ex}")
        return False
    finally:
        shutil.rmtree(hold, ignore_errors=True)


def restart():
    """Start StoryStudio again (used after "Update now" inside the app)."""
    pyw = Path(sys.executable)
    if pyw.name.lower() == "python.exe" and (pyw.parent / "pythonw.exe").exists():
        pyw = pyw.parent / "pythonw.exe"
    subprocess.Popen([str(pyw), str(APP / "StoryStudio.pyw")], cwd=str(APP), **NOWIN)


def _log(msg):
    try:
        from ss_core import log
        log(msg)
    except Exception:
        pass
