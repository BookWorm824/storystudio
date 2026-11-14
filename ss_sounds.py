"""Getting each planned sound: made on this PC (MMAudio), found on Freesound, or written up
as wording for Adobe's generator. Then trimmed, looped, levelled and lined up."""
import json
import math
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from ss_core import (APP_DIR, MODELS, NOWIN, StoryError, ffmpeg, log, log_error, run)
import ss_media as M

SR = M.SR
NEGATIVE = "speech, talking, voice, words, singing, music, melody, distorted, noise, static"
NEGATIVE_MUSIC = "vocals, singing, voice, speech, lyrics, distorted, noise, sound effects"
# a laugh or a sigh IS a voice: only words are unwanted
NEGATIVE_REACTION = "speech, talking, words, singing, music, distorted, noise, static, robotic"
MAX_GEN = 10.0            # seconds per generation (the model is trained on 8 s)

# how loud each kind of sound sits, relative to the narration (LUFS of the sound itself)
DIALOGUE_REF = -18.0
LEVEL_OFFSET = {"background": -19.0, "soft": -15.0, "medium": -12.0, "loud": -9.0}
# the loudest moment (400 ms) of a background may not go above this, relative to the narration
PEAK_CAP = {"background": -13.0}
# spot effects are set by their LOUDEST MOMENT (400 ms), relative to the narration. Setting
# them by their overall loudness made short sounds (a tap, a click) nearly silent and long
# ones (a rustle) louder, so the effects went up and down.
MOMENT_OFFSET = {"soft": -10.0, "medium": -7.0, "loud": -4.0}


def dehash(a):
    """Take the hissy top edge off an AI-made background: AI sound generators often add a
    "hashy" layer of noise up high. A gentle roll-off above about 7 kHz plus a soft dip
    from 4 kHz keeps the birds, wind and room tone but loses the hash."""
    if len(a) < SR // 2:
        return a
    from scipy.signal import butter, sosfilt
    lp = butter(2, 7500, "low", fs=SR, output="sos")
    hp_part = butter(2, 4000, "high", fs=SR, output="sos")
    low = sosfilt(lp, a, axis=0)
    top = sosfilt(hp_part, low, axis=0)
    return (low - top * (1 - 10 ** (-5 / 20))).astype(np.float32)       # about -5 dB up top


def momentary_max(a, rate=None):
    """Loudness of the loudest 400 ms of a sound (LUFS)."""
    rate = rate or SR
    w = int(0.4 * rate)
    if len(a) <= w:
        return M.loudness(a, rate)
    best = -70.0
    for i in range(0, len(a) - w + 1, w // 4):
        best = max(best, M.loudness(a[i:i + w], rate))
    return best


def python_exe():
    """The venv python (the app runs from it; pythonw has no console for the worker)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        exe = exe.with_name("python.exe")
    return str(exe)


def local_ready():
    try:
        import importlib.util
        return (importlib.util.find_spec("mmaudio") is not None and
                importlib.util.find_spec("torch") is not None)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# made on this PC
# ---------------------------------------------------------------------------
def stop_watch(proc, cancel):
    """Stop has to work even while a helper program is silent (downloading its files,
    separating a voice, a long take): check twice a second and end the program. Returns a
    function to call when the program has finished."""
    import threading
    done = threading.Event()
    if cancel:
        def watch():
            while not done.wait(0.5):
                if proc.poll() is not None:
                    return
                if cancel():
                    try:
                        proc.kill()
                    except OSError:
                        pass
                    return
        threading.Thread(target=watch, daemon=True).start()
    return done.set


def has_nvidia():
    """This computer has an NVIDIA graphics card the sound AI can use."""
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=8,
                           **NOWIN)
        return r.returncode == 0 and "GPU" in (r.stdout or "")
    except (OSError, subprocess.SubprocessError):
        return False


def run_local(jobs, progress=None, cancel=None):
    """jobs: dicts for ss_mmaudio_worker (video/duration/prompt/out/seed). Returns
    {index: error text} for jobs that failed."""
    if not jobs:
        return {}
    work = Path(jobs[0]["out"]).parent
    for j in jobs:                                   # never reuse an old file by mistake
        for f in [j["out"]] + list(j.get("cands") or []):
            try:
                Path(f).unlink()
            except OSError:
                pass
    # the graphics card is shared with Premiere and other apps: start with whatever fits,
    # and if it fills up, carry on with a smaller model, then on the processor
    attempts = [(os.environ.get("STORYSOUND_MMAUDIO", "auto"), "auto")]
    no_card = not has_nvidia()
    if no_card:                                      # no graphics card to use at all: the
        attempts = [("small_44k", "cpu")]            # processor, one take per sound
    smaller = {"large_44k_v2": ("medium_44k", "cuda"), "medium_44k": ("small_44k", "cuda"),
               "small_44k": ("small_44k", "cpu")}
    status = {}                                      # index -> "done" | error text
    total = len(jobs)
    back_to_gpu = 0 if no_card else 2                # times it may return to the card
    slow = False                                     # working on the processor right now
    while attempts:
        model, device = attempts.pop(0)
        todo = [i for i in range(total) if i not in status]
        if not todo:
            break
        if device == "cpu":
            # the slow way (about one take a minute): one take per sound instead of the best
            # of several, or a long movie would take many hours
            for i in todo:
                if len(jobs[i].get("cands") or []) > 1:
                    del jobs[i]["cands"][1:]
        spec = work / "pc_jobs.json"
        spec.write_text(json.dumps({"models_dir": str(MODELS / "mmaudio"),
                                    "jobs": [jobs[i] for i in todo], "model": model,
                                    "device": device,
                                    "watch_gpu": device == "cpu" and back_to_gpu > 0}),
                        encoding="utf8")
        errlog = open(work / "pc_worker_log.txt", "a", encoding="utf8")
        proc = subprocess.Popen([python_exe(), str(APP_DIR / "ss_mmaudio_worker.py"),
                                 str(spec)], stdout=subprocess.PIPE, stderr=errlog, text=True,
                                encoding="utf8", errors="replace", **NOWIN)
        full, freed = None, False
        finished = stop_watch(proc, cancel)
        for line in proc.stdout:
            if cancel and cancel():
                proc.kill()
                errlog.close()
                raise StoryError("Stopped.")
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            e = ev.get("event")
            if e == "loading":
                log("pc sound AI loading", ev)
                slow = ev.get("device") == "cpu"
            if e == "loading" and progress:
                where = ("on the graphics card" if ev.get("device") != "cpu" else
                         "on the processor (slower: this computer has no NVIDIA graphics card)"
                         if no_card else
                         "on the processor (slower, because the graphics card is busy)")
                progress(len(status) / total, f"Getting the sound-making AI ready {where}… "
                                              "(the first time it downloads its files)")
            elif e in ("done", "error"):
                i = todo[ev["index"]]
                status[i] = "done" if e == "done" else ev.get("text", "failed")
                if e == "error":
                    log("pc sound failed", ev)
                if progress:
                    progress(len(status) / total,
                             f"Making sounds on this PC: {len(status)} of {total}" + (
                                 (". This computer has no NVIDIA graphics card, so each "
                                  "sound takes about a minute." if no_card else
                                  ". Slow right now: another program (Premiere, Photoshop…) "
                                  "is using the graphics card. It speeds up again when that "
                                  "program is closed.") if slow else ""))
            elif e == "gpu_full":
                full = ev.get("model") or model
                log("graphics card full with", full, ev.get("reason", ""))
            elif e == "gpu_free":
                freed = True
                log("graphics card is free again: back to the fast way")
            elif e == "fatal":
                proc.wait()
                errlog.close()
                raise StoryError(ev.get("text") or "The sound-making AI couldn't start.")
        proc.wait()
        finished()
        errlog.close()
        if cancel and cancel():
            raise StoryError("Stopped.")
        if freed and back_to_gpu > 0:
            back_to_gpu -= 1
            attempts.append(("auto", "auto"))
        elif full:
            nxt = smaller.get(full if full in smaller else "large_44k_v2")
            if nxt and (nxt[1] == "cuda" or device != "cpu"):
                attempts.append(nxt)
    errors = {i: t for i, t in status.items() if t != "done"}
    for i in range(total):
        if i not in status:
            errors[i] = "the sound AI stopped early"
        elif status[i] == "done" and not Path((jobs[i].get("cands") or [jobs[i]["out"]])[0]).exists():
            errors[i] = "no sound came out"
    return errors


def _worker(script, spec_obj, spec_path, on_event, cancel=None):
    """Run one of the helper programs and feed its JSON events to on_event."""
    Path(spec_path).write_text(json.dumps(spec_obj), encoding="utf8")
    errlog = open(Path(spec_path).with_suffix(".log"), "a", encoding="utf8")
    proc = subprocess.Popen([python_exe(), str(APP_DIR / script), str(spec_path)],
                            stdout=subprocess.PIPE, stderr=errlog, text=True, encoding="utf8",
                            errors="replace", **NOWIN)
    finished = stop_watch(proc, cancel)
    try:
        for line in proc.stdout:
            if cancel and cancel():
                proc.kill()
                raise StoryError("Stopped.")
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            on_event(ev)
    finally:
        proc.wait()
        finished()
        errlog.close()
    if cancel and cancel():
        raise StoryError("Stopped.")


# ---------------------------------------------------------------------------
# Stable Audio Open (long, natural background sounds; needs a Hugging Face key)
# ---------------------------------------------------------------------------
SAO_MAX = 47.0


def run_sao(jobs, token, progress=None, cancel=None):
    """Returns (errors {index: text}, problem text or "") ."""
    if not jobs:
        return {}, ""
    work = Path(jobs[0]["out"]).parent
    for j in jobs:                                   # never reuse an old file by mistake
        for f in [j["out"]] + list(j.get("cands") or []):
            try:
                Path(f).unlink()
            except OSError:
                pass
    status, problem = {}, [""]

    def on(ev):
        e = ev.get("event")
        if e == "loading" and progress:
            progress(len(status) / len(jobs), "Getting Stable Audio ready for the background "
                                              "sounds… (the first time it downloads its files)")
        elif e in ("done", "error"):
            status[ev["index"]] = "done" if e == "done" else ev.get("text", "failed")
            if progress:
                progress(len(status) / len(jobs),
                         f"Making background sounds: {len(status)} of {len(jobs)}")
        elif e == "no_access":
            problem[0] = ("Stable Audio said no to your Hugging Face key. Make sure you "
                          "accepted the Stable Audio Open terms on its page while signed in.")
        elif e == "fatal":
            problem[0] = "Stable Audio couldn't start: " + ev.get("text", "")[:200]
            log("sao fatal", ev)
        if e in ("error",):
            log("sao error", ev)
    _worker("ss_sao_worker.py", {"models_dir": str(MODELS), "token": token, "jobs": jobs},
            work / "sao_jobs.json", on, cancel)
    errors = {i: t for i, t in status.items() if t != "done"}
    for i in range(len(jobs)):
        if i not in status:
            errors[i] = problem[0] or "stopped early"
    return errors, problem[0]


# ---------------------------------------------------------------------------
# picking the best take
# ---------------------------------------------------------------------------
# how well a sound matches its description, as the judge scores it: a sound scored against
# a WRONG description lands around 0.0 (three in four under 0.16), a good match 0.3 to 0.7
REDO_BELOW = 0.16             # the best take is no better than a wrong sound: make it again
DROP_BELOW = 0.06             # still nothing like it: switched off rather than put in the movie


def run_pick(groups, progress=None, cancel=None, scores=None):
    """groups: [{prompt, cands, out}]. Copies the best-scoring take of each to out. When
    `scores` (a dict) is given, it is filled with {out: the winning take's match score}."""
    if not groups:
        return {}
    work = Path(groups[0]["out"]).parent
    done = [0]
    flat = {}

    def on(ev):
        if ev.get("event") == "picked":
            done[0] += 1
            if ev.get("flat") is not None:
                flat[groups[ev["index"]]["out"]] = ev["flat"]
            got = [s for s in ev.get("scores") or [] if s is not None]
            if scores is not None and got and not ev.get("error"):
                scores[groups[ev["index"]]["out"]] = max(got)
            if progress:
                progress(done[0] / len(groups),
                         f"Listening to each take and keeping the best: {done[0]} of "
                         f"{len(groups)}")
            if ev.get("error"):
                log("pick problem", ev)
        elif ev.get("event") == "loading" and progress:
            progress(0, "Getting the sound judge ready… (the first time it downloads "
                        "its files)")
    _worker("ss_pick_worker.py", {"models_dir": str(MODELS), "groups": groups},
            work / "pick.json", on, cancel)
    import shutil
    for g in groups:                         # anything the judge missed: keep the first take
        if not Path(g["out"]).exists():
            first = next((c for c in g["cands"] if Path(c).exists()), None)
            if first:
                shutil.copyfile(first, g["out"])
    return flat


def run_score(groups, cancel=None):
    """The judge's score (how well each sound matches its description) for one file per
    group; None where it couldn't tell."""
    if not groups:
        return []
    work = Path(groups[0]["out"]).parent
    scores = [None] * len(groups)

    def on(ev):
        if ev.get("event") == "picked" and ev.get("scores"):
            scores[ev["index"]] = ev["scores"][0]
    _worker("ss_pick_worker.py", {"models_dir": str(MODELS), "groups": groups},
            work / "score.json", on, cancel)
    return scores


# ---------------------------------------------------------------------------
# Freesound
# ---------------------------------------------------------------------------
FS_API = "https://freesound.org/apiv2/search/text/"


def freesound_search(key, words, min_len=0.2, max_len=30.0, avoid=()):
    q = {"query": words, "token": key, "page_size": "15", "sort": "score",
         "fields": "id,name,username,license,duration,previews,avg_rating,num_downloads",
         "filter": f'duration:[{min_len:.1f} TO {max_len:.1f}] '
                   f'license:("Creative Commons 0" OR "Attribution")'}
    url = FS_API + "?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={"User-Agent": "StoryStudio/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf8"))
    except urllib.error.HTTPError as ex:
        if ex.code in (401, 403):
            raise StoryError("Freesound didn't accept your key. Check it in Settings.")
        log("freesound http", ex.code)
        return None
    except Exception:
        log_error("freesound search")
        return None
    res = [x for x in data.get("results", []) if x["id"] not in avoid]
    if not res:
        return None
    # prefer CC0, then popular
    res.sort(key=lambda x: ("Attribution" in x.get("license", "") and
                            "creativecommons.org/publicdomain" not in x.get("license", ""),
                            -(x.get("num_downloads") or 0) ** 0.25 - (x.get("avg_rating") or 0)))
    return res[0]


def freesound_get(hit, out_path):
    url = hit["previews"].get("preview-hq-mp3") or hit["previews"].get("preview-lq-mp3")
    req = urllib.request.Request(url, headers={"User-Agent": "StoryStudio/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(out_path, "wb") as f:
        f.write(r.read())
    lic = hit.get("license", "")
    credit = ""
    if "publicdomain" not in lic and "zero" not in lic.lower():
        credit = f'"{hit["name"]}" by {hit["username"]} (freesound.org/s/{hit["id"]}) – CC BY'
    return out_path, credit


# ---------------------------------------------------------------------------
# shaping a raw sound into what gets placed on the timeline
# ---------------------------------------------------------------------------
def load48(src):
    """Any audio file -> (n, 2) float32 at 48 kHz."""
    r = subprocess.run([ffmpeg(), "-v", "error", "-i", str(src), "-ac", "2", "-ar", str(SR),
                        "-f", "f32le", "-"], capture_output=True, **NOWIN)
    a = np.frombuffer(r.stdout[: len(r.stdout) // 8 * 8], dtype=np.float32)
    return a.reshape(-1, 2).copy()


def onset(a, frac=0.12):
    """Index where the sound really starts (first sample above frac of its peak)."""
    env = np.abs(a).max(axis=1)
    pk = env.max() if len(env) else 0
    if pk <= 0:
        return 0
    idx = np.argmax(env > pk * frac)
    return max(0, int(idx) - int(0.01 * SR))


def lead_silence(a, thresh_db=-55.0):
    """Index where real silence at the start of a sound ends (0 if it starts straight away).
    Unlike onset(), quiet content before a louder moment is kept."""
    env = np.abs(a).max(axis=1)
    nz = np.nonzero(env > 10 ** (thresh_db / 20))[0]
    if len(nz) == 0:
        return 0
    return max(0, int(nz[0]) - int(0.01 * SR))


def trim_tail(a, thresh_db=-55.0):
    env = np.abs(a).max(axis=1)
    lim = 10 ** (thresh_db / 20)
    nz = np.nonzero(env > lim)[0]
    if len(nz) == 0:
        return a[:1]
    return a[: min(len(a), nz[-1] + int(0.05 * SR))]


def loop_to(a, length, xfade=1.5):
    """Repeat a (with crossfades) until it is `length` samples long."""
    if len(a) >= length:
        return a[:length]
    if len(a) < SR // 20:                    # nothing to repeat: silence, never an endless loop
        return np.zeros((max(0, length), 2), dtype=np.float32)
    xf = int(min(xfade * SR, len(a) * 0.3))
    out = a.copy()
    ramp = np.linspace(0, 1, xf, dtype=np.float32)[:, None]
    while len(out) < length:
        nxt = a
        head = out[:-xf] if xf else out
        mixz = out[-xf:] * (1 - ramp) + nxt[:xf] * ramp if xf else np.zeros((0, 2), np.float32)
        out = np.concatenate([head, mixz, nxt[xf:]])
    return out[:length]


def fades(a, fin=0.01, fout=0.1, smooth=False):
    """Fade in/out. smooth=True uses equal-power curves, so two backgrounds crossfading
    over each other keep a steady loudness (no dip in the middle)."""
    a = a.copy()
    n = len(a)
    fi, fo = min(int(fin * SR), n // 2), min(int(fout * SR), n // 2)
    if fi > 0:
        r = np.linspace(0, 1, fi, dtype=np.float32)
        a[:fi] *= (np.sin(r * np.pi / 2) if smooth else r)[:, None]
    if fo > 0:
        r = np.linspace(1, 0, fo, dtype=np.float32)
        a[-fo:] *= (np.sin(r * np.pi / 2) if smooth else r)[:, None]
    return a


# backgrounds overlap the next place by this much on each side of a scene change and
# crossfade (seconds); music blends more gently
BED_OVERLAP = {"ambience": 0.8, "music": 1.6}


def target_lufs(level, movie_type="", kind=""):
    if kind == "music":                          # a music bed sits well under the narration
        return DIALOGUE_REF - 11.0
    t = DIALOGUE_REF + LEVEL_OFFSET.get(level, -6.0)
    if movie_type == "music video":
        t -= 4.0
    return t


def normalise(a, lufs_target, level=None):
    """Gain the sound so it measures lufs_target, without its loudest moment jumping out
    (PEAK_CAP) and with a sample-peak ceiling."""
    if len(a) == 0:
        return a, 0.0
    L = M.loudness(a)
    if L <= -69:
        return a, 0.0
    g = lufs_target - L
    if level in MOMENT_OFFSET:               # spot effect: by its loudest moment
        # lufs_target already carries any music-video adjustment
        g = lufs_target + (MOMENT_OFFSET[level] - LEVEL_OFFSET[level]) - momentary_max(a)
    elif level in PEAK_CAP:
        cap = DIALOGUE_REF + PEAK_CAP[level]
        g = min(g, cap - momentary_max(a))
    peak = float(np.abs(a).max()) or 1.0
    g = min(g, 20 * math.log10(0.89 / peak))       # keep peaks under -1 dBFS
    return (a * 10 ** (g / 20)).astype(np.float32), g


def write_wav(a, path):
    import soundfile as sf
    sf.write(str(path), a, SR, subtype="PCM_24")
    return str(path)


def best_hit_time(motion, mfps, t0, t1):
    """The moment of biggest on-screen change between t0 and t1 (for lining up a hit)."""
    i0, i1 = max(0, int(t0 * mfps)), min(len(motion), int(t1 * mfps) + 1)
    if i1 - i0 < 2:
        return None
    seg = motion[i0:i1]
    if seg.max() < 0.015:                        # nothing really moves: leave it
        return None
    return (i0 + int(np.argmax(seg))) / mfps
