"""ChatGPT inside StoryStudio, through OpenAI's Codex program on this PC.

For people who don't have Claude. Codex is OpenAI's own program: signed in with a ChatGPT
account it uses that ChatGPT plan (no API key, nothing extra to pay). StoryStudio runs it in
an empty folder with read-only access, so it only answers questions; pictures (stills of the
movie) are attached as images. Same interface as ss_claude.ask.
"""
import copy
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ss_core import APP_DATA, NOWIN, StoryError, log

MODEL = os.environ.get("STORYSTUDIO_CHATGPT_MODEL", "")       # empty: the plan's default
CONSOLE = 0x00000010                                          # CREATE_NEW_CONSOLE


_store = {}


def _store_codex():
    """codex.exe inside the Codex desktop app from the Microsoft Store. Its folder name
    changes with every app update, so ask Windows where the app is (once per run)."""
    if "exe" in _store:
        return _store["exe"]
    _store["exe"] = None
    if os.name != "nt":
        return None
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-AppxPackage -Name 'OpenAI.Codex' | Sort-Object Version "
                            "-Descending | Select-Object -First 1).InstallLocation"],
                           capture_output=True, text=True, timeout=30, **NOWIN)
        loc = (r.stdout or "").strip()
        if loc:
            p = Path(loc) / "app" / "resources" / "codex.exe"
            if p.exists():
                _store["exe"] = str(p)
    except Exception:
        pass
    return _store["exe"]


def find():
    exe = os.environ.get("STORYSTUDIO_CODEX") or shutil.which("codex")
    if exe:
        return exe
    la = Path(os.environ.get("LOCALAPPDATA", ""))
    ad = Path(os.environ.get("APPDATA", ""))
    for p in (la / "Programs" / "OpenAI" / "Codex" / "bin" / "codex.exe",
              la / "Microsoft" / "WinGet" / "Links" / "codex.exe",
              ad / "npm" / "codex.cmd",
              Path.home() / ".local" / "bin" / "codex"):
        if str(p) not in (".", "") and p.exists():
            return str(p)
    return _store_codex()


def _env():
    env = dict(os.environ)
    for k in ("OPENAI_API_KEY", "CODEX_API_KEY"):     # always the ChatGPT plan, never a paid key
        env.pop(k, None)
    return env


def install(progress=None):
    """OpenAI's own installer (per-user, no admin rights)."""
    say = progress or (lambda f, m: None)
    if os.name != "nt":
        raise StoryError("Install Codex from chatgpt.com/codex, then try again.")
    say(0.1, "Installing Codex, OpenAI's program that lets StoryStudio talk to ChatGPT…")
    env = dict(os.environ, CODEX_NON_INTERACTIVE="1")
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
                        "irm https://chatgpt.com/codex/install.ps1 | iex"],
                       capture_output=True, text=True, timeout=900, env=env, **NOWIN)
    if not find():
        log("codex install failed", (r.stderr or r.stdout)[-800:])
        raise StoryError("Codex didn't install. Check the internet connection and try again.")
    say(1.0, "Codex is installed.")


def signed_in():
    exe = find()
    if not exe:
        return False
    try:
        r = subprocess.run([exe, "login", "status"], capture_output=True, text=True,
                           timeout=30, env=_env(), **NOWIN)
    except Exception:
        return False
    out = (r.stdout + r.stderr).lower()
    return r.returncode == 0 and "not logged in" not in out


def open_sign_in():
    """A small window where Codex opens the ChatGPT sign-in page in the browser."""
    exe = find()
    if not exe:
        raise StoryError("Codex isn't installed yet.")
    if os.name == "nt":
        # one plain command line for cmd: a list would have Python escape the quotes round the
        # program path as \" and cmd then can't find it (the window just flashed shut)
        subprocess.Popen("cmd /c \"title Sign in to ChatGPT & echo Sign in with your "
                         "ChatGPT account in the web page that opens. & echo When it says you "
                         "are signed in, close this window and go back to StoryStudio. & echo."
                         f" & \"{exe}\" login & pause\"", env=_env(),
                         creationflags=CONSOLE)
    else:
        subprocess.Popen([exe, "login"], env=_env())


def status():
    if not find():
        return "missing"
    return "ready" if signed_in() else "signed_out"


def strict(schema):
    """OpenAI's structured output wants every field required and no extra fields."""
    s = copy.deepcopy(schema)

    def fix(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)
    fix(s)
    return s


def ask(prompt, schema, system=None, images=None, timeout=1200):
    """One request to ChatGPT; returns the structured answer (a dict).
    images: [(label, path_to_jpg), ...], attached in that order."""
    exe = find()
    if not exe:
        raise StoryError("ChatGPT isn't set up yet. Open Settings in StoryStudio to set it up.")
    work = APP_DATA / "chatgpt"
    work.mkdir(parents=True, exist_ok=True)
    empty = work / "empty"                      # nothing in here for it to look at
    empty.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=work))
    sf, of = tmp / "schema.json", tmp / "answer.json"
    sf.write_text(json.dumps(strict(schema)), encoding="utf8")
    full = ((system + "\n\n") if system else "") + prompt + (
        "\n\nAnswer only from what is given here (the text and the attached pictures). Don't "
        "run any commands, open files or search the web.")
    common = ["--skip-git-repo-check", "--output-schema", str(sf), "-o", str(of), "--json"]
    if images:
        full = ("The pictures attached are, in order: " +
                "; ".join(f"{k + 1}) {label}" for k, (label, _) in enumerate(images)) +
                ".\n\n" + full)
        for _, path in images:
            common.append(f"--image={path}")       # one per flag, so nothing after is swallowed
    if MODEL:
        common += ["-m", MODEL]
    cmd = [exe, "exec", "-s", "read-only", "-C", str(empty), "--color", "never", *common,
           "--ephemeral", "-"]
    try:
        r = subprocess.run(cmd, input=full, capture_output=True, text=True, encoding="utf8",
                           errors="replace", timeout=timeout, cwd=str(empty), env=_env(),
                           **NOWIN)
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp, ignore_errors=True)
        raise StoryError("ChatGPT took too long to answer. Please try again.")
    errs = []
    for line in (r.stdout or "").splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") in ("error", "turn.failed"):
            errs.append(str(ev.get("message") or (ev.get("error") or {}).get("message") or ""))
    try:
        answer = json.loads(of.read_text("utf8"))
    except (OSError, ValueError):
        answer = None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not isinstance(answer, dict):
        msg = (" ".join(errs) or r.stderr or r.stdout or "")[-600:]
        low = msg.lower()
        log("chatgpt failed:", msg)
        if any(k in low for k in ("not logged in", "login", "unauthorized", "401", "sign in")):
            raise StoryError("ChatGPT isn't signed in. Open Settings in StoryStudio and sign "
                             "in, then try again.")
        if any(k in low for k in ("usage limit", "rate limit", "limit reached", "quota", "429")):
            raise StoryError("You've reached your ChatGPT usage limit for now. It resets after "
                             "a while.")
        raise StoryError("ChatGPT couldn't answer this time. Please try again.")
    return answer
