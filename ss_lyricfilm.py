"""The "Elegant" lyrics look: a pretty lyric video, not karaoke.

Each line appears as it is sung, word by word (every word fades in and rises into place when
it is sung), in fine serif lettering with a soft glow, one word picked out larger in a script
face. A line sits where the picture under it is calmest (a quiet corner or band, away from the
busy part of the frame), drifts gently upward while it shows, and fades away when it is done.

    render(st, lines, out_mov, fps, settings, progress, cancel)   the see-through lyrics video
    sample(st, settings, out)                                      one still showing the look
"""
import math
import subprocess
from pathlib import Path

from ss_core import NOWIN, StoryError, ffmpeg, fmt_time, log, run

MAIN_FONTS = ("pala.ttf", "constan.ttf", "georgia.ttf", "cambria.ttc", "times.ttf")
ACCENT_FONTS = ("MTCORSVA.TTF", "mtcorsva.ttf", "palai.ttf", "constani.ttf", "georgiai.ttf")
INK = (255, 246, 230)            # warm ivory words
ACCENT = (255, 206, 128)         # soft gold: the picked-out word and the glow
SIZE = 0.064                     # main lettering, as a share of the picture's short side
ACCENT_SCALE = 1.8
STEP = 1.3                       # the second row steps in this many letter-heights, like a poem
ACCENT_ROOM = 0.9                # extra space (in spaces) each side of the picked-out word
REVEAL = 0.45                    # seconds a word takes to fade in and rise
ACCENT_REVEAL = 0.7
EARLY = 0.12                     # a word starts appearing this much before it is sung
HOLD = 1.6                       # a line stays this long after its last word
FADE = 0.55                      # and fades away over this long
DRIFT = 5.0                      # pixels per second the line floats up (at 1080 high)
SMALL = {"A", "AN", "THE", "OF", "IN", "TO", "AND", "OR", "BUT", "WHEN", "LIKE", "MY",
         "YOUR", "THAT", "WITH", "FOR", "ON", "AT", "IS", "IF", "SO", "AS", "FROM", "INTO",
         "I", "YOU", "ME", "WE", "IT", "BE", "ALL", "ARE", "WAS", "OUR", "THIS", "JUST"}

# where a line may sit: (alignment, x of its anchor, y of its middle), as shares of the frame
SPOTS_WIDE = [("left", 0.07, 0.79), ("center", 0.50, 0.83), ("right", 0.93, 0.79),
              ("left", 0.07, 0.21), ("center", 0.50, 0.16), ("right", 0.93, 0.21)]
SPOTS_TALL = [("center", 0.50, 0.80), ("center", 0.50, 0.68), ("center", 0.50, 0.20)]
HALO_FROM = 0.33                 # a picture lighter than this behind the words: a dark halo


def halo_of(light):
    return min(1.0, max(0.0, (light - HALO_FROM) / 0.25))


LOW_BONUS = 0.06                 # lower spots are a little preferred (where eyes expect words)
STAY_BONUS = 0.05                # and staying in the same spot as the line before


def _hex(v):
    import re
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(v or "").strip())
    return tuple(int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4)) if m else None


def colours(settings):
    s = settings or {}
    return _hex(s.get("karaoke_color")) or INK, _hex(s.get("karaoke_highlight")) or ACCENT


def _font(size, names):
    from PIL import ImageFont
    for name in names:
        try:
            return ImageFont.truetype(name, max(8, int(size)))
        except OSError:
            continue
    return ImageFont.load_default()


def _norm(text):
    return "".join(ch for ch in text.upper() if ch.isalpha() or ch == "'")


# ---------------------------------------------------------------------------
# lines -> cards (what shows together)
# ---------------------------------------------------------------------------
def cards_of(lines, chars):
    """Each lyric line becomes one card of one or two rows (a long line is split at a natural
    place; a very long one becomes several cards)."""
    from ss_karaoke import short_lines, for_screen
    out = []
    for line in lines:
        ws = [w for w in line.get("words") or [] if w.get("text")]
        if not ws:
            continue
        rows = [r["words"] for r in for_screen(short_lines([dict(line, words=ws)], chars))]
        if not rows:
            continue
        for k in range(0, len(rows), 2):
            out.append({"rows": rows[k:k + 2]})
    for c in out:
        words = [w for r in c["rows"] for w in r]
        c["start"], c["end"] = words[0]["start"], words[-1]["end"]
        best, idx = None, None                     # the word picked out: the longest real word
        for i, w in enumerate(words):
            n = _norm(w["text"])
            if n in SMALL or len(n) < 4:
                continue
            if best is None or len(n) >= best:
                best, idx = len(n), i
        c["accent"] = idx
    for i, c in enumerate(out):                    # when each card shows
        c["appear"] = max(0.0, c["start"] - 0.15)
        if i:
            c["appear"] = max(c["appear"], out[i - 1]["gone"])
        c["gone"] = c["end"] + HOLD + FADE
        if i + 1 < len(out):
            c["gone"] = min(c["gone"], max(c["end"] + 0.3, out[i + 1]["start"] - 0.2))
    return out


# ---------------------------------------------------------------------------
# drawing a card
# ---------------------------------------------------------------------------
def _word_image(text, font, ink, glow, strong, halo=0.0):
    """One word with its soft shadow and glow. Returns (image, origin, advance): drawing the
    word with its baseline-left at (x, y) means pasting the image at (x - ox, y - oy)."""
    from PIL import Image, ImageDraw, ImageFilter
    size = font.size
    l, t, r, b = font.getbbox(text, anchor="ls")
    pad = int(size * 0.5) + 6
    w, h = (r - l) + 2 * pad, (b - t) + 2 * pad
    ox, oy = pad - l, pad - t
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = max(1, size // 28)
    if halo > 0:                                   # a light picture behind: a soft dark halo
        hl = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(hl).text((ox, oy), text, font=font, anchor="ls",
                                fill=(0, 0, 0, int(160 * halo)),
                                stroke_width=max(2, size // 8),
                                stroke_fill=(0, 0, 0, int(160 * halo)))
        im.alpha_composite(hl.filter(ImageFilter.GaussianBlur(max(4, size * 0.3))))
    ImageDraw.Draw(sh).text((ox + d, oy + d * 2), text, font=font, anchor="ls",
                            fill=(0, 0, 0, min(255, int(165 + 80 * halo))))
    sh = sh.filter(ImageFilter.GaussianBlur(max(2, size * 0.09)))
    im.alpha_composite(sh)
    gl = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(gl).text((ox, oy), text, font=font, anchor="ls",
                            fill=glow + (150 if strong else 85,),
                            stroke_width=max(1, size // 16), stroke_fill=glow + (150 if strong else 85,))
    gl = gl.filter(ImageFilter.GaussianBlur(max(3, size * 0.2)))
    im.alpha_composite(gl)
    if strong:
        im.alpha_composite(gl)
    ImageDraw.Draw(im).text((ox, oy), text, font=font, anchor="ls", fill=ink + (255,))
    return im, (ox, oy), font.getlength(text)


def build_card(card, W, H, ink, accent, halo=0.0):
    """The card's words drawn, and where each one sits relative to the card's top-left."""
    base = min(W, H) * SIZE
    for _ in range(8):
        main = _font(base, MAIN_FONTS)
        acc = _font(base * ACCENT_SCALE, ACCENT_FONTS)
        space = main.getlength(" ")
        rows, k = [], 0
        for row in card["rows"]:
            items = []
            for w in row:
                strong = k == card["accent"]
                f = acc if strong else main
                img, origin, adv = _word_image(w["text"], f, accent if strong else ink, accent,
                                               strong, halo)
                items.append({"img": img, "origin": origin, "adv": adv, "w": w,
                              "strong": strong, "k": k})
                k += 1
            room = space * ACCENT_ROOM             # the script word's swashes need more room
            for a, b in zip(items, items[1:]):
                a["gap"] = space + (room if a["strong"] or b["strong"] else 0)
            items[-1]["gap"] = 0
            width = sum(i["adv"] + i["gap"] for i in items)
            up = main.getmetrics()[0]
            for i in items:                        # the picked-out word may stand taller
                if i["strong"]:
                    up = max(up, -acc.getbbox(i["w"]["text"], anchor="ls")[1])
            rows.append({"items": items, "width": width, "up": up,
                         "down": main.getmetrics()[1]})
        maxw = max(r["width"] for r in rows)
        if maxw <= W * 0.84 or base < min(W, H) * 0.03:
            break
        base *= 0.9
    gap = base * 0.28
    y = 0.0
    for r in rows:                                 # baselines, one row under the other
        y += r["up"]
        r["base"] = y
        y += r["down"] + gap
    return {"rows": rows, "w": maxw, "h": y - gap, "size": base}


def place(built, spot, W, H):
    """Each word's position (baseline-left) on the frame for a card at this spot."""
    align, ax, ay = spot
    top = H * ay - built["h"] / 2
    top = min(max(top, H * 0.05), H * 0.95 - built["h"])
    out = []
    for n, r in enumerate(built["rows"]):
        step = built["size"] * STEP * n           # the second row steps in, like a poem
        if align == "left":
            x = W * ax + step
        elif align == "right":
            x = W * ax - r["width"] - step
        else:
            x = W * ax - r["width"] / 2
        x = min(max(x, W * 0.04), W * 0.96 - r["width"])
        for i in r["items"]:
            out.append((i, x, top + r["base"]))
            x += i["adv"] + i["gap"]
    return out, (W * ax - (built["w"] if align == "right" else built["w"] / 2
                           if align == "center" else 0), top, built["w"], built["h"])


# ---------------------------------------------------------------------------
# where the picture is calm
# ---------------------------------------------------------------------------
def _frame(pic, t, work, k):
    """A small grey picture of the movie at time t, as a numpy array (or None)."""
    import numpy as np
    from PIL import Image
    if not pic or not Path(pic).exists():
        return None
    f = Path(work) / f"lyricspot_{k:04d}.png"
    try:
        run([ffmpeg(), "-y", "-v", "error", "-ss", f"{max(0.0, t):.2f}", "-i", str(pic),
             "-frames:v", "1", "-vf", "scale=320:-2", str(f)])
        a = np.asarray(Image.open(f).convert("L"), dtype=np.float32) / 255.0
        f.unlink()
        return a
    except Exception:
        return None


def _cost(a, box, W, H):
    """How busy and how bright the picture is inside a box (shares of 0..1)."""
    import numpy as np
    h, w = a.shape
    x, y, bw, bh = box
    x0, x1 = int(max(0, x / W * w)), int(min(w, (x + bw) / W * w)) + 1
    y0, y1 = int(max(0, y / H * h)), int(min(h, (y + bh) / H * h)) + 1
    reg = a[y0:y1, x0:x1]
    if reg.size < 4:
        return 9.0
    gx = np.abs(np.diff(reg, axis=1)).mean() if reg.shape[1] > 1 else 0
    gy = np.abs(np.diff(reg, axis=0)).mean() if reg.shape[0] > 1 else 0
    edges = (gx + gy) * 6
    bright = max(0.0, float(reg.mean()) - 0.55) * 1.4      # light words want a darker place
    return float(edges + bright + reg.std() * 0.6)


def choose_spots(cards, built, pic, W, H, work):
    """The calmest spot for each card (lower ones a little preferred, and keeping the spot of
    the card before when it is nearly as good, so the eye isn't sent all over the frame)."""
    spots = SPOTS_WIDE if W >= H else SPOTS_TALL
    prev, out = None, []
    for k, (c, b) in enumerate(zip(cards, built)):
        a = _frame(pic, (c["start"] + c["end"]) / 2, work, k)
        best = None
        for s in spots:
            _, box = place(b, s, W, H)
            cost = _cost(a, box, W, H) if a is not None else 0.0
            if s[2] < 0.5:
                cost += LOW_BONUS
            if s != prev:
                cost += STAY_BONUS
            if best is None or cost < best[0]:
                best = (cost, s)
        light = 0.0
        if a is not None:
            _, box = place(b, best[1], W, H)
            hh, ww = a.shape
            reg = a[int(max(0, box[1] / H * hh)):int(min(hh, (box[1] + box[3]) / H * hh)) + 1,
                    int(max(0, box[0] / W * ww)):int(min(ww, (box[0] + box[2]) / W * ww)) + 1]
            light = float(reg.mean()) if reg.size else 0.0
        out.append((best[1], light))
        prev = best[1]
    return out


# ---------------------------------------------------------------------------
# the video
# ---------------------------------------------------------------------------
def _ease(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def _faded(item, q, cache):
    """The word image at q/16 of its full strength (kept, since words reuse the same steps)."""
    key = (id(item["img"]), q)
    im = cache.get(key)
    if im is None:
        im = item["img"].copy()
        a = im.getchannel("A").point(lambda v: v * q // 16)
        im.putalpha(a)
        cache[key] = im
        if len(cache) > 4000:
            cache.clear()
    return im


def render(st, lines, out_mov, fps, settings=None, progress=None, cancel=None):
    from PIL import Image
    W, H = int(st.get("width") or 1920), int(st.get("height") or 1080)
    ink, accent = colours(settings)
    cards = cards_of(lines, 26 if W >= H else 16)
    say = progress or (lambda f, m: None)
    say(0.0, "Designing the lyrics…")
    built = [build_card(c, W, H, ink, accent) for c in cards]
    pic = st.get("video") or st.get("picture") or st.get("export")
    work = Path(st.get("work") or Path(out_mov).parent)
    work.mkdir(parents=True, exist_ok=True)
    chosen = choose_spots(cards, built, pic, W, H, work)
    spots = [c[0] for c in chosen]
    for k, (_, light) in enumerate(chosen):        # on a light picture the words get a halo
        if light > HALO_FROM:
            built[k] = build_card(cards[k], W, H, ink, accent, halo_of(light))
    placed = [place(b, s, W, H)[0] for b, s in zip(built, spots)]
    st.setdefault("karaoke", {})["shown"] = len(cards)
    st["karaoke"]["spots"] = [s[0] + (" low" if s[2] > 0.5 else " high") for s in spots]
    dur = float(st["duration"])
    n = int(math.ceil(dur * fps))
    rise = 16 * H / 1080
    drift = DRIFT * H / 1080
    cmd = [ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s",
           f"{W}x{H}", "-r", f"{fps:.6f}", "-i", "-", "-c:v", "qtrle", "-pix_fmt", "argb",
           str(out_mov)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, **NOWIN)
    blank = bytes(W * H * 4)
    last_key, last = None, blank
    cache = {}
    ci = 0
    try:
        for f in range(n):
            if cancel and f % 50 == 0 and cancel():
                raise StoryError("Stopped.")
            t = f / fps
            while ci < len(cards) and cards[ci]["gone"] <= t:
                ci += 1
            showing = [k for k in range(max(0, ci - 1), min(len(cards), ci + 2))
                       if cards[k]["appear"] <= t < cards[k]["gone"]]
            parts = []
            for k in showing:
                c = cards[k]
                life = min(1.0, (c["gone"] - t) / FADE)
                up = int((t - c["appear"]) * drift + (1 - life) * rise * 0.6)
                for item, x, y in placed[k]:
                    w = item["w"]
                    r = (t - (w["start"] - EARLY)) / (ACCENT_REVEAL if item["strong"] else REVEAL)
                    if r <= 0:
                        continue
                    e = _ease(r)
                    q = int(round(16 * e * life))
                    if q <= 0:
                        continue
                    dy = int((1 - e) * rise)
                    parts.append((k, item["k"], q, dy - up, x, y, item))
            key = tuple(p_[:4] for p_ in parts)
            if not parts:
                data = blank
            elif key == last_key:
                data = last
            else:
                canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                for k, _, q, dy, x, y, item in parts:
                    im = _faded(item, q, cache)
                    ox, oy = item["origin"]
                    canvas.alpha_composite(im, (int(x - ox), int(y - oy + dy)))
                data = canvas.tobytes()
            last_key, last = key, data
            p.stdin.write(data)
            if f % 120 == 0:
                say(0.05 + 0.95 * f / max(1, n), f"Drawing the lyrics: {fmt_time(t)} of "
                    f"{fmt_time(dur)}")
    finally:
        try:
            p.stdin.close()
        except OSError:
            pass
        err = p.stderr.read().decode("utf8", "replace")
        p.wait()
    if p.returncode != 0:
        log("lyric film encode failed", err[-1500:])
        raise StoryError("StoryStudio couldn't make the lyrics video.")
    log("lyric film:", len(cards), "lines;", " ".join(st["karaoke"]["spots"]))
    return str(out_mov)


def sample(st, settings, out, bg):
    """bg: the picture (RGBA, full size) to draw a sample line on."""
    W, H = bg.size
    ink, accent = colours(settings)
    words = "Twinkle twinkle little star, how I wonder what you are".split()
    ws = [{"text": w, "norm": _norm(w), "start": i * 0.4, "end": i * 0.4 + 0.35}
          for i, w in enumerate(words)]
    card = cards_of([{"words": ws, "start": 0, "end": ws[-1]["end"]}], 26 if W >= H else 16)[0]
    b = build_card(card, W, H, ink, accent)
    import numpy as np
    a = np.asarray(bg.convert("L").resize((320, max(1, int(320 * H / W)))),
                   dtype=np.float32) / 255.0
    spots = SPOTS_WIDE if W >= H else SPOTS_TALL
    spot = min(spots, key=lambda s: _cost(a, place(b, s, W, H)[1], W, H) +
               (LOW_BONUS if s[2] < 0.5 else 0))
    _, box = place(b, spot, W, H)
    hh, ww = a.shape
    reg = a[int(max(0, box[1] / H * hh)):int(min(hh, (box[1] + box[3]) / H * hh)) + 1,
            int(max(0, box[0] / W * ww)):int(min(ww, (box[0] + box[2]) / W * ww)) + 1]
    if reg.size and reg.mean() > HALO_FROM:
        b = build_card(card, W, H, ink, accent, halo_of(float(reg.mean())))
    for item, x, y in place(b, spot, W, H)[0]:
        ox, oy = item["origin"]
        bg.alpha_composite(item["img"], (int(x - ox), int(y - oy)))
    bg.convert("RGB").resize((640, int(640 * H / W))).save(out, quality=88)
    return str(out)
