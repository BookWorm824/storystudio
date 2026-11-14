"""Karaoke lyrics: paste the song's lyrics; every word lights up as it is sung.

    lines = parse_lyrics(text)
    time_lyrics(st, lines, progress)      # times every word to the singing (on the timeline)
    render_overlay(st, out_mov)           # a see-through video: words fill with colour
    karaoke(st, settings, text, progress) # all of it + preview movie + Premiere project copy
"""
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

from ss_core import APP_DIR, MODELS, NOWIN, StoryError, ffmpeg, fmt_time, log, run, safe_name
import ss_media as M

SECTION = re.compile(r"^\s*\[.*\]\s*$")                # [Chorus], [anything]
# a line in round brackets is a label only when it reads like one: "(Verse 2)", "(Chorus x2)".
# Otherwise it is sung: "(Shine, shine)", "(Ooh) I love you (ooh)".
PAREN_LABEL = re.compile(r"^\s*\(\s*(verse|chorus|pre[- ]?chorus|post[- ]?chorus|bridge|intro|"
                         r"outro|hook|refrain|interlude|instrumental|solo|break|tag|ending|"
                         r"spoken|repeat|x\s*\d)(?![a-z])[^()]*\)\s*$", re.I)
# "x2" at the end of a line, on its own or in brackets ("max 2" is not a repeat)
REPEAT = re.compile(r"(?:^|(?<=[\s\[(]))\s*[\[(]?\s*x\s*(\d)\s*[\])]?\s*$", re.I)
ONES = ("ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN ELEVEN TWELVE THIRTEEN FOURTEEN "
        "FIFTEEN SIXTEEN SEVENTEEN EIGHTEEN NINETEEN").split()
TENS = "TWENTY THIRTY FORTY FIFTY SIXTY SEVENTY EIGHTY NINETY".split()


def number_words(n):
    """How a number is sung, as letters for lining it up with the singing: 5 -> FIVE."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10 - 2] + (ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return ONES[n // 100] + "HUNDRED" + (number_words(n % 100) if n % 100 else "")
    return "".join(ONES[int(d)] for d in str(n))


def is_label(s):
    return bool(SECTION.match(s) or PAREN_LABEL.match(s))


def _section_key(label):
    s = re.sub(r"[\[\]()]", " ", label.lower())
    s = REPEAT.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def _expand_sections(text):
    """Section labels are not sung, but they often stand in for words: a label like [Chorus]
    with nothing under it means "sing the chorus again", and [Chorus x2] means twice. Returns
    the lyric lines in the order they're sung."""
    sections, cur = [], None
    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s:
            continue
        if is_label(s):
            m = REPEAT.search(s.rstrip("]) "))
            cur = {"key": _section_key(s), "times": int(m.group(1)) if m else 1, "lines": []}
            sections.append(cur)
        else:
            if cur is None:
                cur = {"key": "", "times": 1, "lines": []}
                sections.append(cur)
            cur["lines"].append(s)
    seen, out = {}, []
    for sec in sections:
        body = sec["lines"]
        if body:
            if sec["key"]:
                seen[sec["key"]] = body
        elif sec["key"]:
            # same label as before, or the same kind ("Chorus" for "Final Chorus")
            body = seen.get(sec["key"])
            if body is None:
                kind = next((w for w in ("chorus", "hook", "refrain", "pre-chorus", "bridge")
                             if w in sec["key"]), None)
                if kind:
                    body = next((b for k, b in reversed(list(seen.items())) if kind in k), None)
            body = body or []
        out += body * max(1, min(4, sec["times"]))
    return out


def parse_lyrics(text):
    """Lyrics in the order they are sung -> lines of words. Section labels like [Chorus]
    are not shown, but an empty [Chorus] repeats the chorus and [Chorus x2] repeats it twice;
    a line ending in (x2) is sung twice."""
    lines = []
    for s in _expand_sections(text):
        times = 1
        m = REPEAT.search(s)
        if m:
            times = max(1, min(4, int(m.group(1))))
            s = s[:m.start()].strip()
        words = []
        for w in s.split():
            norm = re.sub(r"[^A-Za-z']", "", w).upper().strip("'")
            digits = re.sub(r"\D", "", w)
            if not norm and digits:                        # "5 little ducks", "1, 2, 3"
                norm = number_words(int(digits[:6]))
            if norm:
                words.append({"text": w, "norm": norm})
            elif words:                                    # "-", "&" ... stays with a word
                words[-1]["text"] += " " + w
        for _ in range(times):
            if words:
                lines.append({"text": s, "words": [dict(x) for x in words]})
    return lines


def song_audio(st, work):
    """The movie's sound (the song) on the timeline, as one wav."""
    import storysound as E
    seq = E._sequence(st)
    clips = st.get("clips") or M.build_stems(seq, work)
    out = Path(work) / "song.wav"
    M.mix([dict(c, gain_db=c.get("gain_db", 0.0)) for c in clips], seq["duration"], out)
    return str(out)


def time_lyrics(st, lines, progress=None, cancel=None):
    work = Path(st["work"]) / "karaoke"
    work.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(0.05, "Getting the song from your timeline…")
    song = song_audio(st, work)
    try:                                   # no sound means the song file wasn't found
        import numpy as _np
        if float(_np.abs(M.wav_mono16k(song)).max()) < 0.01:
            raise StoryError("StoryStudio couldn't hear the song on your timeline. The song "
                             "file may have been moved or renamed; open the project in "
                             "Premiere, relink the song, save, and try again.")
    except StoryError:
        raise
    except Exception:
        pass
    words = [w["norm"] for l in lines for w in l["words"]]
    if not words:
        raise StoryError("Please paste the song's lyrics first.")
    job = {"models_dir": str(MODELS), "ffmpeg": ffmpeg(), "audio": song, "words": words,
           "out": str(work / "timing.json"), "vocals_out": str(work / "vocals.wav"),
           "lines": [len(l["words"]) for l in lines]}
    (work / "job.json").write_text(json.dumps(job), encoding="utf8")
    try:                                     # never read an earlier run's timings by mistake
        (work / "timing.json").unlink()
    except OSError:
        pass
    import ss_sounds as S
    proc = subprocess.Popen([S.python_exe(), str(APP_DIR / "ss_karaoke_worker.py"),
                             str(work / "job.json")], stdout=subprocess.PIPE,
                            stderr=open(work / "worker_log.txt", "w", encoding="utf8"),
                            text=True, encoding="utf8", errors="replace", **NOWIN)
    fatal = ""
    finished = S.stop_watch(proc, cancel)
    for line in proc.stdout:
        if cancel and cancel():
            proc.kill()
            raise StoryError("Stopped.")
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        e = ev.get("event")
        if progress and e == "separating":
            progress(0.2, "Separating the singing from the music…")
        elif progress and e == "aligning":
            progress(0.55, "Lining every word up with the singing…")
        elif e == "fatal":
            fatal = ev.get("text", "")
    proc.wait()
    finished()
    if cancel and cancel():
        raise StoryError("Stopped.")
    if fatal or not (work / "timing.json").exists():
        log("karaoke worker failed", fatal)
        low = fatal.lower()
        if "out of memory" in low:
            raise StoryError("The graphics card ran out of room while lining up the lyrics. "
                             "Close Premiere or other big programs and try again.")
        if any(w in low for w in ("connection", "urlopen", "resolve", "timed out", "download",
                                  "modulenotfound", "no module")):
            raise StoryError("StoryStudio couldn't get the files it needs to line up the "
                             "lyrics. Check the internet connection and try again.")
        raise StoryError("StoryStudio couldn't line the lyrics up with the song. Check that "
                         "the lyrics are in the order they're sung.")
    result = json.loads((work / "timing.json").read_text(encoding="utf8"))
    # the worker returns the lines in the order they're sung: a section sung more times than
    # it's written (an extra chorus) is repeated, and lines written but not sung there are
    # left out rather than flashed on screen at the wrong time
    import copy
    timed = []
    for item in result["lines"]:
        line = copy.deepcopy(lines[item["line"]])
        for w, t in zip(line["words"], item["words"]):
            if t:
                w.update(start=t["start"], end=max(t["end"], t["start"] + 0.12),
                         score=t["score"])
        timed.append(line)
    left_out = [lines[i]["text"] for i in result.get("dropped_lines") or []]
    repeats = [a["lines"] for a in result.get("added") or []]
    lines = timed
    if not lines:
        raise StoryError("StoryStudio couldn't hear these lyrics in the song.")
    _fill_gaps(lines)
    for l in lines:
        l["start"] = l["words"][0]["start"]
        l["end"] = l["words"][-1]["end"]
    weak = sum(1 for l in lines for w in l["words"] if w.get("score", 1) < 0.15)
    total = sum(len(l["words"]) for l in lines)
    st["karaoke"] = {"lines": lines, "weak": weak, "total": total, "left_out": left_out,
                     "repeats": repeats}
    if left_out:
        log("karaoke: left out (not sung where written):", left_out)
    if repeats:
        log("karaoke: added repeated sections:", repeats)
    if progress:
        progress(0.75, f"Timed {total} words.")
    return lines


def _fill_gaps(lines):
    """Words the aligner couldn't place get times between their neighbours."""
    flat = [w for l in lines for w in l["words"]]
    known = [i for i, w in enumerate(flat) if "start" in w]
    if not known:
        raise StoryError("StoryStudio couldn't hear the lyrics in the song.")
    for i, w in enumerate(flat):
        if "start" in w:
            continue
        prev = max([j for j in known if j < i], default=None)
        nxt = min([j for j in known if j > i], default=None)
        a = flat[prev]["end"] if prev is not None else flat[nxt]["start"] - 0.5
        b = flat[nxt]["start"] if nxt is not None else a + 0.5
        n = (nxt if nxt is not None else len(flat)) - (prev if prev is not None else -1)
        pos = i - (prev if prev is not None else -1)
        s = a + (b - a) * (pos - 0.5) / n
        w.update(start=s, end=s + max(0.12, (b - a) / n * 0.8), score=0.0)


# ---------------------------------------------------------------------------
# drawing: classic two-row karaoke; each word fills with colour as it is sung
# ---------------------------------------------------------------------------
STYLES = {
    # big, bold and gold, with a strong outline: easy for children to follow
    "kids": {"fonts": ("seguibl.ttf", "segoeuib.ttf", "arialbd.ttf"), "size": 0.085,
             "min": 0.045, "base": (255, 255, 255, 255), "fill": (255, 196, 64, 255),
             "outline": (20, 14, 30, 255), "stroke": 14, "min_stroke": 2, "shadow": (0, 0, 0, 140),
             "blur": 3, "drop": 3, "chars": (24, 16),
             "rows": {2: ((0.745, 0.875), (0.70, 0.78)), 1: ((0.86,), (0.76,))}},
    # finer serif letters, smaller and lower; sung words turn bright with a soft gold glow
    "elegant": {"fonts": ("constan.ttf", "georgia.ttf", "pala.ttf", "times.ttf"),
                "size": 0.062, "min": 0.034, "base": (255, 255, 255, 115),
                "fill": (255, 248, 232, 255), "glow": (255, 196, 90, 235), "outline": (0, 0, 0, 80), "stroke": 45, "min_stroke": 1,
                "shadow": (0, 0, 0, 200), "blur": 6, "drop": 2, "chars": (34, 22),
                "rows": {2: ((0.80, 0.885), (0.74, 0.80)), 1: ((0.885,), (0.79,))}},
}
STYLE = STYLES["kids"]


def style_of(settings):
    s = STYLES.get((settings or {}).get("karaoke_style"), STYLES["kids"])
    try:
        rows = 1 if int((settings or {}).get("karaoke_rows", 2)) == 1 else 2
    except (TypeError, ValueError):
        rows = 2
    return s, rows


def _font(size, names=STYLE["fonts"]):
    from PIL import ImageFont
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _line_images(line, W, H, style=None):
    """The line drawn twice (before and while it's sung), plus where each word sits."""
    from PIL import Image, ImageDraw, ImageFilter
    S = style or STYLE
    size = int(min(W, H) * S["size"])
    text = " ".join(w["text"] for w in line["words"])
    while True:
        font = _font(size, S["fonts"])
        tw = font.getlength(text)
        if tw <= W * 0.9 or size <= int(min(W, H) * S["min"]):
            break
        size -= 2
    stroke = max(S["min_stroke"], size // S["stroke"])
    asc, desc = font.getmetrics()
    pad = S["blur"] * 2 + (size // 5 if S.get("glow") else 0)
    h = asc + desc + stroke * 2 + 8 + pad
    w_img = int(tw) + stroke * 2 + 8 + pad
    xs, x = [], stroke + 4
    for i, wd in enumerate(line["words"]):
        piece = wd["text"] + (" " if i < len(line["words"]) - 1 else "")
        wl = font.getlength(wd["text"])
        xs.append((x, x + wl))
        x += font.getlength(piece)
    imgs = {}
    for key in ("base", "fill"):
        im = Image.new("RGBA", (w_img, h), (0, 0, 0, 0))
        sh = Image.new("RGBA", (w_img, h), (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((stroke + 4 + S["drop"], stroke + S["drop"]), text,
                                font=font, fill=S["shadow"], stroke_width=stroke,
                                stroke_fill=S["shadow"])
        sh = sh.filter(ImageFilter.GaussianBlur(S["blur"]))
        im.alpha_composite(sh)
        if key == "fill" and S.get("glow"):           # a soft golden glow behind sung words
            gl = Image.new("RGBA", (w_img, h), (0, 0, 0, 0))
            ImageDraw.Draw(gl).text((stroke + 4, stroke), text, font=font, fill=S["glow"],
                                    stroke_width=stroke + max(2, size // 20),
                                    stroke_fill=S["glow"])
            gl = gl.filter(ImageFilter.GaussianBlur(max(4, size // 9)))
            im.alpha_composite(gl)
            im.alpha_composite(gl)
        ImageDraw.Draw(im).text((stroke + 4, stroke), text, font=font, fill=S[key],
                                stroke_width=stroke, stroke_fill=S["outline"])
        imgs[key] = im
    return imgs["base"], imgs["fill"], xs


PUNCT = re.compile(r"[,;:.!?—–-]['\")]*$")
SMALL = {"A", "AN", "THE", "OF", "IN", "TO", "AND", "OR", "BUT", "WHEN", "LIKE", "MY",
         "YOUR", "THAT", "WITH", "FOR", "ON", "AT", "IS", "IF", "SO", "AS", "FROM", "INTO"}


def short_lines(lines, max_chars=24):
    """Long lyric lines split into short ones that are easy for children to follow.
    Breaks go after commas and full stops and where the singer pauses, never leaving one
    little word on its own if it can be helped."""
    target = max_chars * 0.8
    out = []
    for l in lines:
        ws = l["words"]
        n = len(ws)
        if n == 0:
            continue
        if len(" ".join(w["text"] for w in ws)) <= max_chars:
            out.append(l)
            continue

        def cost(i, j):
            c = len(" ".join(w["text"] for w in ws[i:j]))
            if c > max_chars + 2 and j - i > 1:
                return math.inf
            return (c - target) ** 2 / 10 + 25 + (60 if j - i == 1 else 0)

        def brk(i):                                   # breaking just before word i
            p = 0 if PUNCT.search(ws[i - 1]["text"]) else 30
            if ws[i - 1]["norm"] in SMALL:
                p += 25                               # don't end a line on "the", "in", "when"
            if ws[i].get("start", 0) - ws[i - 1].get("end", 0) > 0.8:
                p -= 40                               # the singer takes a breath here
            return p

        best = [0.0] + [math.inf] * n
        back = [0] * (n + 1)
        for j in range(1, n + 1):
            for i in range(j):
                c = best[i] + cost(i, j) + (brk(i) if i else 0)
                if c < best[j]:
                    best[j], back[j] = c, i
        cuts, j = [], n
        while j > 0:
            cuts.append((back[j], j))
            j = back[j]
        for i, j in reversed(cuts):
            part = ws[i:j]
            out.append({"text": " ".join(w["text"] for w in part), "words": part,
                        "start": part[0]["start"], "end": part[-1]["end"]})
    return out


LEAD = 0.5          # seconds each pair of lines shows before its first word is sung


def _visibility(lines, rows=2):
    """When each line appears and disappears, and which row it uses.
    Lines come in pages of two, read top to bottom like a book: the top line is sung, then
    the bottom one, then both change together. A row only ever holds one line, so words
    are never drawn on top of words. With one row, every line is a page of its own."""
    pages, i, n = [], 0, len(lines)
    while i < n:
        if rows == 2 and i + 1 < n and lines[i + 1]["start"] - lines[i]["end"] < 2.5:
            pages.append([i, i + 1])
            i += 2
        else:
            pages.append([i])
            i += 1
    vis = [None] * n
    prev_gone = 0.0
    for p, page in enumerate(pages):
        first, last = lines[page[0]], lines[page[-1]]
        appear = max(prev_gone, first["start"] - LEAD)
        appear = max(0.0, min(appear, first["start"] - 0.05))
        gone = last["end"] + 1.5
        if p + 1 < len(pages):
            nxt = lines[pages[p + 1][0]]["start"]
            gone = min(gone, max(last["end"] + 0.05, nxt - LEAD))
        for row, k in enumerate(page):
            vis[k] = (appear, gone, row)
        prev_gone = gone
    return vis


def _layout(st, settings):
    W, H = int(st.get("width") or 1920), int(st.get("height") or 1080)
    S, n_rows = style_of(settings)
    wide = W >= H
    rows = tuple(H * y for y in S["rows"][n_rows][0 if wide else 1])
    return W, H, S, n_rows, rows, S["chars"][0 if wide else 1]


def sample(st, settings, out):
    """One still showing how the lyrics will look (a frame of the movie if there is one)."""
    from PIL import Image
    W, H, S, n_rows, rows, chars = _layout(st, settings)
    bg = None
    work = Path(st["work"]) if st.get("work") else None
    pic = next((p for p in (st.get("picture"), st.get("export"),
                            work and work / "picture.mp4") if p and Path(p).exists()), None)
    if pic:                                            # a frame of the movie
        try:
            t = float(st.get("duration") or 20) * 0.4
            frame = Path(out).with_suffix(".frame.jpg")
            run([ffmpeg(), "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(pic),
                 "-frames:v", "1", "-vf", f"scale={W}:{H}", str(frame)])
            bg = Image.open(frame).convert("RGBA")
        except Exception:
            bg = None
    if bg is None and st.get("thumb") and Path(st["thumb"]).exists():   # the project card
        try:
            bg = Image.open(st["thumb"]).convert("RGBA").resize((W, H))
        except Exception:
            bg = None
    if bg is None:
        bg = Image.new("RGBA", (W, H), (38, 52, 84, 255))
    words = "Twinkle twinkle little star how I wonder what you are".split()
    lines = []
    for k, chunk in enumerate((words[:4], words[4:])):
        ws = [{"text": w, "start": k * 3 + i * 0.5, "end": k * 3 + i * 0.5 + 0.45}
              for i, w in enumerate(chunk)]
        lines.append({"words": ws})
    for row, line in enumerate(lines[:n_rows]):
        base, fill, xs = _line_images(line, W, H, S)
        layer = base.copy()
        if row == 0:                                   # the first line half sung
            cut = int(xs[1][1])
            layer.paste(fill.crop((0, 0, cut, fill.height)), (0, 0))
        bg.alpha_composite(layer, ((W - layer.width) // 2, int(rows[row] - layer.height / 2)))
    bg.convert("RGB").resize((640, int(640 * H / W))).save(out, quality=85)
    return str(out)


def render_overlay(st, out_mov, fps=None, progress=None, cancel=None, settings=None):
    """A video the length of the movie with a see-through background and the lyrics."""
    from PIL import Image
    W, H, S, n_rows, rows, chars = _layout(st, settings)
    lines = short_lines(st["karaoke"]["lines"], chars)
    st["karaoke"]["shown"] = len(lines)
    fps = fps or float(st.get("fps") or 24.0)
    dur = float(st["duration"])
    imgs = [_line_images(l, W, H, S) for l in lines]
    vis = _visibility(lines, n_rows)
    n = int(math.ceil(dur * fps))
    cmd = [ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s",
           f"{W}x{H}", "-r", f"{fps:.6f}", "-i", "-", "-c:v", "qtrle", "-pix_fmt", "argb",
           str(out_mov)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, **NOWIN)
    blank = bytes(W * H * 4)
    last_key, last_bytes = None, blank
    try:
        for f in range(n):
            if cancel and cancel() and f % 50 == 0:
                raise StoryError("Stopped.")
            t = f / fps
            shown = []
            for i, (a, b, row) in enumerate(vis):
                if a <= t < b:
                    shown.append(i)
            if not shown:
                key = None
                data = blank
            else:
                # what changes the picture: which lines, how far the fill is, the fades
                parts = []
                for i in shown:
                    a, b, row = vis[i]
                    fade = min(1.0, (t - a) / 0.25, (b - t) / 0.25)
                    fill_x = _fill_x(lines[i], imgs[i][2], t)
                    parts.append((i, round(fade, 2), int(fill_x)))
                key = tuple(parts)
                if key == last_key:
                    data = last_bytes
                else:
                    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                    for i, fade, fill_x in parts:
                        base, fill, xs = imgs[i]
                        a, b, row = vis[i]
                        layer = base.copy()
                        if fill_x > 0:
                            layer.paste(fill.crop((0, 0, fill_x, fill.height)), (0, 0))
                        if fade < 1:
                            alpha = layer.getchannel("A").point(lambda v: int(v * fade))
                            layer.putalpha(alpha)
                        x0 = (W - layer.width) // 2
                        y0 = int(rows[row] - layer.height / 2)
                        canvas.alpha_composite(layer, (x0, y0))
                    data = canvas.tobytes()
            last_key, last_bytes = key, data
            p.stdin.write(data)
            if progress and f % 120 == 0:
                progress(f / n, f"Drawing the lyrics: {fmt_time(t)} of {fmt_time(dur)}")
    finally:
        p.stdin.close()
        err = p.stderr.read().decode("utf8", "replace")
        p.wait()
    if p.returncode != 0:
        log("overlay encode failed", err[-1500:])
        raise StoryError("StoryStudio couldn't make the lyrics video.")
    return str(out_mov)


def _fill_x(line, xs, t):
    """How far (in pixels) the colour has filled the line at time t."""
    fill = 0.0
    for (x0, x1), w in zip(xs, line["words"]):
        if t >= w["end"]:
            fill = x1
        elif t > w["start"]:
            fill = x0 + (x1 - x0) * (t - w["start"]) / max(0.05, w["end"] - w["start"])
            break
        else:
            break
    return fill


# ---------------------------------------------------------------------------
# all of it
# ---------------------------------------------------------------------------
def karaoke(st, settings, lyrics_text, progress=None, cancel=None):
    import storysound as E
    say = progress or (lambda f, m: None)
    lines = parse_lyrics(lyrics_text)
    if not lines:
        raise StoryError("Please paste the song's lyrics first.")
    st["mode"] = "karaoke"
    seq = E._sequence(st)
    st.setdefault("width", seq["width"])
    st.setdefault("height", seq["height"])
    if not st.get("picture") or not Path(st["picture"]).exists():
        if st.get("export") and Path(st["export"]).exists():
            st["picture"] = st["export"]
        else:
            say(0.01, "Getting the picture ready for the preview…")
            st["picture"] = M.build_picture(seq, Path(st["work"]), lambda f: say(
                0.01 + 0.1 * f, "Getting the picture ready for the preview…"))
    time_lyrics(st, lines, lambda f, m: say(0.12 + 0.4 * f, m), cancel)
    od = E.out_dir(st)
    name = safe_name(st["seq_name"])
    overlay = od / f"{name} - karaoke lyrics.mov"
    render_overlay(st, overlay, seq["fps"], lambda f, m: say(0.52 + 0.3 * f, m), cancel,
                   settings)
    say(0.83, "Making the preview…")
    preview = od / f"{name} - karaoke (preview).mp4"
    song = Path(st["work"]) / "karaoke" / "song.wav"
    from ss_core import probe
    pi = probe(st["picture"])
    pw, ph = pi.get("width") or 640, pi.get("height") or 360
    run([ffmpeg(), "-y", "-v", "error", "-i", st["picture"], "-i", overlay, "-i", str(song),
         "-filter_complex", f"[1:v]scale={pw}:{ph}[ov];[0:v][ov]overlay=0:0:format=auto[v]",
         "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-c:a", "aac", "-b:a", "256k", "-shortest", str(preview)])
    out = {"preview": str(preview), "folder": str(od), "overlay": str(overlay)}
    say(0.93, "Adding the lyrics to a copy of your Premiere project…")
    import ss_prwrite as W
    real = Path(st["project"])
    import storysound as _E
    proj = _E.copy_path(real, "karaoke")
    try:
        out["project"] = W.add_overlay(st.get("snapshot") or st["project"], st["seq_uid"],
                                       overlay, 0.0, st["duration"], st["width"],
                                       st["height"], seq["fps"], proj)
    except StoryError as ex:
        out["project_error"] = str(ex)
    except Exception:
        from ss_core import log_error
        log_error("karaoke project")
        out["project_error"] = ("StoryStudio couldn't add the lyrics to the project copy. The "
                                "lyrics video is in the StoryStudio folder; drag it onto the "
                                "top video track at the very start.")
    k = st["karaoke"]
    st["levels"] = [f"Timed {k['total']} words to the singing."]
    if k["weak"]:
        st["levels"].append(f"{k['weak']} word(s) were hard to hear; check them in the preview.")
    for n in k.get("repeats") or []:
        st["levels"].append(f"A section of {n} lines (like the chorus) is sung more times than "
                            "it's written in the lyrics, so StoryStudio added the repeat.")
    gone = k.get("left_out") or []
    if gone:
        st["levels"].append(
            f"{len(gone)} line(s) didn't seem to be sung where they're written, so they're left "
            "out instead of showing at the wrong time: " + "; ".join(f"“{t}”" for t in gone[:4])
            + ("…" if len(gone) > 4 else "") + ". If they are sung, check the lyrics order.")
    lst = od / f"{name} - lyrics timing.txt"
    text = "\n".join(f"{fmt_time(l['start'])}  {l['text']}" for l in k["lines"])
    if gone:
        text += ("\n\nLeft out (didn't seem to be sung where they're written):\n" +
                 "\n".join(f"  {t}" for t in gone))
    lst.write_text(text, encoding="utf8")
    out["list"] = str(lst)
    st["outputs"] = out
    st["stage"] = "karaoke"
    E.save(st)
    say(1.0, "Done!")
    return out
