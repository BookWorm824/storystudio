"""Joining a folder of separate voice files into one narration, in script order.

    join(folder, shots, out, progress, cancel) -> (heard, seconds, info)

Many stories keep their voices as one file per line ("Narrator Line 7.wav", "Benny - carrots
taste good.wav"), or a long narration plus the characters' lines as their own files.
StoryStudio listens to every file, finds where its words are in the script, cuts a long file
where another voice's line belongs in the middle of it, and writes one sound file with every
piece in script order. Silence and stray noise before and after the words are left behind.
The originals are never changed.

`heard` is every word of the joined file with its time, which is what the clips are timed to.
"""
import difflib
import re
import subprocess
import wave
from pathlib import Path

import numpy as np

from ss_core import NOWIN, StoryError, ffmpeg, log, log_error
import ss_media as M

SOUNDS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".aif", ".aiff"}
COPY = re.compile(r"\((?:alt|copy)?\s*\d+\)|\bcopy\b|\balt\s*\d+\b", re.I)
SERIES = re.compile(r"^(.*?)\s*(\d+)$")
JOINED = "(joined)"
RATE = 48000
SHORT = 6                       # a file of this many words or fewer is one short line
LINE = 25                       # up to this many words a file is "a line", not a long reading
GAP = 0.35                      # the pause put between two pieces, in seconds
BEFORE, AFTER = 0.2, 0.35       # how much sound is kept around a piece's first and last word
QUIET = 2.5                     # a silence longer than this inside a file is closed up


def tokens(text):
    """The words of a text the way they are compared: "Ding-ding! Don't" -> ding ding dont."""
    return [t.replace("'", "") for t in re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))
            if t.strip("'")]


def _natural(p):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name.lower())]


def voice_files(folder):
    """The sound files in a folder, without copies of a file that is also there ("(alt 2)",
    "(2)") and without a narration StoryStudio joined earlier."""
    folder = Path(folder)
    if not folder.is_dir():
        raise StoryError("That folder couldn't be opened.")
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SOUNDS
             and JOINED not in p.name.lower()]
    plain = {COPY.sub("", p.stem).strip().lower() for p in files if not COPY.search(p.stem)}
    return sorted((p for p in files if not COPY.search(p.stem) or
                   COPY.sub("", p.stem).strip().lower() not in plain), key=_natural)


def _short_claim(script, owned, words):
    """Where a short line is in the script: the first place, not yet taken, that matches it
    best. Returns {script index: word index} or {}."""
    n = len(words)
    best, where = 0, None
    for i in range(0, max(1, len(script) - n + 1)):
        part = script[i:i + n + 1]
        blocks = difflib.SequenceMatcher(None, part, words, autojunk=False) \
            .get_matching_blocks()
        got = {i + b.a + k: b.b + k for b in blocks for k in range(b.size)}
        got = {a: b for a, b in got.items() if a not in owned}
        if len(got) > best:
            best, where = len(got), got
    need = n if n == 1 else max(2, -(-n * 6 // 10))      # one word: exactly; else 60%
    return where if where and best >= need else {}


def _long_claim(script, owned, words):
    """Where a longer recording is in the script, wherever it runs alongside it. Small
    chance matches far from anything else are dropped."""
    sm = difflib.SequenceMatcher(None, script, words, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size]
    keep = [b.size >= 3 for b in blocks]
    changed = True
    while changed:                                       # small blocks next to a kept one
        changed = False
        for x, b in enumerate(blocks):
            if keep[x]:
                continue
            for y in (x - 1, x + 1):
                if 0 <= y < len(blocks) and keep[y]:
                    o = blocks[y]
                    near_a = min(abs(b.a - (o.a + o.size)), abs(o.a - (b.a + b.size)))
                    near_b = min(abs(b.b - (o.b + o.size)), abs(o.b - (b.b + b.size)))
                    if near_a <= 4 and near_b <= 4:
                        keep[x] = changed = True
                        break
    return {b.a + k: b.b + k for x, b in enumerate(blocks) if keep[x]
            for k in range(b.size) if (b.a + k) not in owned}


def _span(info, k0, k1):
    """The seconds of a file that hold its words k0..k1, with a little air around them."""
    w = info["words"]
    a = w[k0]["s"] - (BEFORE if k0 == 0 else min(BEFORE, max(0.0, w[k0]["s"] - w[k0 - 1]["e"]) / 2))
    b = w[k1]["e"] + (AFTER if k1 == len(w) - 1 else
                      min(AFTER, max(0.0, w[k1 + 1]["s"] - w[k1]["e"]) / 2))
    a = max(0.0, a)
    return a, min(info["seconds"], max(b, a + 0.1))


def plan(script, files):
    """Which piece of which file goes where. `script` is the story's spoken words in order
    (from tokens()); `files` is [{"name", "words": [{"t","s","e"}], "seconds", "norm"}].
    Returns (pieces in script order, {file number: why it wasn't used}, script words found).
    A piece is {"file", "a", "b" (seconds of the file), "k0", "k1" (its words), "at"}."""
    owned = {}                                           # script index -> (file no, word no)
    size = [len(f["norm"]) for f in files]
    # Single lines choose first, the longest of them first (so "what matters most" can't take
    # its words out of the middle of a longer line that also says them); long readings take
    # what is left, so a character's own file wins over a narrator reading the same words.
    order = sorted((f for f in range(len(files)) if size[f] <= LINE), key=lambda f: -size[f]) \
        + sorted((f for f in range(len(files)) if size[f] > LINE), key=lambda f: size[f])
    unused = {}

    def claim_of(f, taken):
        words = [t for t, _ in files[f]["norm"]]
        return (_short_claim if len(words) <= SHORT else _long_claim)(script, taken, words)
    for f in order:
        if not size[f]:
            unused[f] = "silent"
            continue
        got = claim_of(f, owned)
        if not got:
            unused[f] = "double" if claim_of(f, {}) else "lost"
        for i, k in got.items():
            owned[i] = (f, files[f]["norm"][k][1])       # the word's place in the file
    for f in [f for f, why in unused.items() if why == "double" and size[f] >= 3]:
        got = claim_of(f, {})                            # a line that only a much longer
        holders = {owned[i][0] for i in got if i in owned}   # file is holding: it takes it
        if holders and all(size[h] >= 2 * size[f] for h in holders):
            for i, k in got.items():
                owned[i] = (f, files[f]["norm"][k][1])
            del unused[f]
    pieces = []
    for f, info in enumerate(files):
        mine = sorted((k, i) for i, (g, k) in owned.items() if g == f)
        if not mine:
            continue
        w = info["words"]
        runs, cur = [], [mine[0]]
        for (k0, i0), (k1, i1) in zip(mine, mine[1:]):
            lo, hi = sorted((i0, i1))
            other = any(owned.get(i, (f,))[0] != f for i in range(lo + 1, hi))
            quiet = k1 > k0 and max(w[k]["s"] - w[k - 1]["e"] for k in range(k0 + 1, k1 + 1)) > QUIET
            if other or i1 < i0 or quiet:                # another voice, or a long silence
                runs.append(cur)
                cur = []
            cur.append((k1, i1))
        runs.append(cur)
        for run in runs:
            k0, k1 = run[0][0], run[-1][0]
            a, b = _span(info, k0, k1)
            pieces.append({"file": f, "k0": k0, "k1": k1, "a": a, "b": b,
                           "at": min(i for _, i in run), "to": max(i for _, i in run)})
    # a numbered file whose words weren't found ("Narrator Line 20") goes after number 19
    names = {}
    for f, info in enumerate(files):
        m = SERIES.match(Path(info["name"]).stem)
        names[f] = (m.group(1).strip().lower(), int(m.group(2))) if m else \
            (Path(info["name"]).stem.strip().lower(), 1)
    for f in sorted(unused, key=lambda f: names[f][1]):
        if unused[f] != "lost" or not files[f]["words"]:
            continue
        before = [p for p in pieces if names[p["file"]][0] == names[f][0] and
                  names[p["file"]][1] == names[f][1] - 1]
        if not before:
            continue
        a, b = _span(files[f], 0, len(files[f]["words"]) - 1)
        lo = max(p["to"] for p in before)
        after = [p["at"] for p in pieces if names[p["file"]][0] == names[f][0] and
                 names[p["file"]][1] == names[f][1] + 1]
        window = script[int(lo) + 1: int(min(after)) if after else len(script)]
        words = [t for t, _ in files[f]["norm"]]
        best = max(difflib.SequenceMatcher(None, window, words, autojunk=False)
                   .get_matching_blocks(), key=lambda m: m.size)
        # between its neighbours, where even one of its words is; else right after no. n-1
        at = int(lo) + 1 + best.a - best.b - 0.25 if best.size else lo + 0.5
        pieces.append({"file": f, "k0": 0, "k1": len(files[f]["words"]) - 1, "a": a, "b": b,
                       "at": at, "to": at})
        del unused[f]
    pieces.sort(key=lambda p: p["at"])
    return pieces, unused, len(owned)


def _sound(path, a, b):
    """A piece of a file as 48 kHz stereo 16-bit samples, with soft edges."""
    r = subprocess.run([ffmpeg(), "-v", "error", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}",
                        "-i", str(path), "-ac", "2", "-ar", str(RATE), "-f", "s16le", "-"],
                       capture_output=True, **NOWIN)
    x = np.frombuffer(r.stdout[: len(r.stdout) // 4 * 4], dtype=np.int16).reshape(-1, 2)
    x = x.astype(np.float32)
    n = min(int(RATE * 0.01), len(x) // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x.astype(np.int16)


def listen(folder, progress=None, cancel=None):
    """Every voice file in the folder with the words heard in it."""
    say = progress or (lambda f, m: None)
    paths = voice_files(folder)
    if not paths:
        raise StoryError("There are no sound files in that folder.")
    rec = M.speech_recognizer()
    files = []
    for n, p in enumerate(paths):
        if cancel and cancel():
            raise StoryError("Stopped.")
        say(n / len(paths), f"Listening to your voice files ({n + 1} of {len(paths)})…")
        try:
            mono = M.wav_mono16k(p)
            words = sorted(M.transcribe_raw(mono, rec=rec), key=lambda w: w["s"])
        except Exception:
            log_error("voices: listening to " + p.name)
            mono, words = np.zeros(0, dtype=np.float32), []
        files.append({"path": str(p), "name": p.name, "words": tidy(words),
                      "seconds": len(mono) / 16000.0})
    return files


def tidy(words):
    """Word ends are sometimes far too late when silence follows a word: bring them back to
    a sensible length, and never past the next word."""
    for k, w in enumerate(words):
        e = min(w["e"], w["s"] + 0.3 + 0.1 * len(w["t"]))
        if k + 1 < len(words):
            e = min(e, max(w["s"] + 0.05, words[k + 1]["s"]))
        w["e"] = round(max(e, w["s"] + 0.05), 2)
    return words


def join(folder, shots, out, progress=None, cancel=None, files=None):
    """Listen to every voice file in the folder, put the pieces in script order and write
    them as one file. Returns (heard words with their times in that file, its length, info)."""
    say = progress or (lambda f, m: None)
    script = [k for s in shots for k in tokens(s["words"])]
    if not script:
        raise StoryError("The voice files can only be put in order once StoryStudio knows the "
                         "script's words.")
    files = files or listen(folder, lambda f, m: say(0.7 * f, m), cancel)
    for f in files:
        tidy(f["words"])
        f["norm"] = [(t, k) for k, w in enumerate(f["words"]) for t in tokens(w["t"])]
    pieces, unused, found = plan(script, files)
    if not pieces:
        raise StoryError("None of those sound files matched the script. Is it the right "
                         "folder?")
    say(0.72, "Joining the voices in script order…")
    heard, t = [], 0.3
    out = Path(out)
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(np.zeros((int(t * RATE), 2), dtype=np.int16).tobytes())
        for n, p in enumerate(pieces):
            if cancel and cancel():
                raise StoryError("Stopped.")
            f = files[p["file"]]
            x = _sound(f["path"], p["a"], p["b"])
            wf.writeframes(x.tobytes())
            for w in f["words"][p["k0"]: p["k1"] + 1]:
                heard.append({"t": w["t"], "s": round(t + w["s"] - p["a"], 2),
                              "e": round(t + w["e"] - p["a"], 2)})
            p["start"] = round(t, 2)
            t += len(x) / RATE
            wf.writeframes(np.zeros((int(GAP * RATE), 2), dtype=np.int16).tobytes())
            t += int(GAP * RATE) / RATE
            say(0.72 + 0.28 * (n + 1) / len(pieces), "Joining the voices in script order…")
        wf.writeframes(np.zeros((int(0.7 * RATE), 2), dtype=np.int16).tobytes())
        t += 0.7
    used = {p["file"] for p in pieces}
    info = {"file": str(out), "name": out.name, "files": len(used), "pieces": len(pieces),
            "found": round(found / len(script), 2),
            "doubles": [files[f]["name"] for f, why in unused.items() if why == "double"],
            "left_out": [files[f]["name"] for f, why in unused.items() if why != "double"],
            "order": [{"name": files[p["file"]]["name"], "at": p["start"],
                       "from": round(p["a"], 2), "to": round(p["b"], 2),
                       "says": " ".join(w["t"] for w in
                                        files[p["file"]]["words"][p["k0"]: p["k1"] + 1])[:80]}
                      for p in pieces]}
    log("voices:", len(files), "files,", len(pieces), "pieces,",
        f"{info['found']:.0%} of the script; doubles", info["doubles"], "left out",
        info["left_out"])
    return heard, round(t, 2), info
