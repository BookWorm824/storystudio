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
# a second address for the same files (a free copy of GitHub): used when GitHub's own address
# can't be reached from this computer (slow connection, or a filter that blocks it)
MIRROR = f"https://cdn.jsdelivr.net/gh/{REPO}@main/"
LAST = {"error": "", "base": None}           # why the last check failed, for Settings
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


def _why(ex):
    s = f"{type(ex).__name__}: {ex}"
    low = s.lower()
    if "certificate" in low or "ssl" in low:
        return "a security program on this computer blocked the secure connection"
    if "timed out" in low or "timeout" in low:
        return "the connection was too slow"
    if "getaddrinfo" in low or "name or service" in low or "nodename" in low:
        return "no internet connection (or the address is blocked)"
    if "403" in low or "forbidden" in low:
        return "the address is blocked"
    return s[:160]


def check(timeout=8, patient=False):
    """The newer version's info ({"version", "files", "notes"}) or None. Tries GitHub, then the
    second address; with patient=True each gets a second, longer try. Why it failed is logged
    and kept in LAST["error"] (Settings shows it)."""
    errors = []
    sources = [RAW] if "STORYSTUDIO_UPDATE_URL" in os.environ else [RAW, MIRROR]
    for base in sources:
        for t in ((timeout, max(20, timeout * 2)) if patient else (timeout,)):
            try:
                info = json.loads(_get("version.json", t, base).decode("utf8"))
            except Exception as ex:
                errors.append(f"{base.split('/')[2]}: {_why(ex)}")
                continue
            LAST["error"], LAST["base"] = "", base
            if _vt(info.get("version")) > _vt(local_version()):
                return info
            return None
    _log("update check failed: " + "; ".join(errors))
    reasons = [e.split(": ", 1)[-1] for e in errors]
    LAST["error"] = reasons[-1] if len(set(reasons)) == 1 else         "no internet connection, or this computer blocks the update address"
    return None


def apply(info, progress=None):
    """Download and install the new version. Returns True when installed."""
    base = _pinned_base()
    if not base and LAST.get("base") == MIRROR:   # GitHub can't be reached: the second address
        base = MIRROR
    if base and base != MIRROR:
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
        got = json.loads((hold / "version.json").read_text(encoding="utf8"))
        if _vt(got.get("version")) < _vt(info.get("version")):
            raise ValueError("the files downloaded weren't all from the new version yet")
        for f in files:                      # everything checked: now swap them in
            if f != "version.json":
                shutil.copy2(hold / f, APP / f)
        shutil.copy2(hold / "version.json", APP / "version.json")   # last, so a failed
        return True                                                  # swap retries next time
    except Exception as ex:
        LAST["error"] = f"the download stopped ({_why(ex)})"
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
