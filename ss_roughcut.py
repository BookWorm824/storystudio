"""A rough cut for Premiere, built from the pictures and clips made for a shot list.

    build(r, folder=None) -> {"file", "name", "shots", "seconds", "stand_ins", "missing", ...}

Writes "<Title> - rough cut.xml" into the folder with the pictures and clips. It is the kind
of file Premiere reads with File > Import: one timeline with every shot in order, each clip
cut to the seconds the clip check called good and set to the speed it gave. A shot with no
good clip yet is stood in for by its picture (coloured differently, with a marker), so the
whole story can be watched from start to end. Clips that were never checked go in whole, so
this also works with no AI at all. Nothing the user made is changed.
"""
import time
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape

from ss_core import StoryError, log, log_error, probe, safe_name
import ss_clips as K

HOLD = 4.0                      # seconds for a stand-in picture when the shot has no length
STILL_FRAMES = 1080000          # how long Premiere may stretch a picture (frames)
STILL_IN = 90000
STAND_IN_COLOUR = "Mango"
UNCHECKED_COLOUR = "Lavender"


def _rate(fps):
    """(timebase, ntsc) the way the file wants a frame rate: 23.976 -> (24, True)."""
    for tb in (24, 30, 60):
        if abs(fps - tb * 1000 / 1001) < 0.02:
            return tb, True
    return max(1, int(round(fps or 24))), False


def _real(tb, ntsc):
    return tb * 1000 / 1001 if ntsc else float(tb)


def _usable(f):
    c = f.get("check") or {}
    return f["kind"] == "clip" and c.get("verdict") in ("keep", "use part") and \
        (f.get("seconds") or 0) > 0.3


def _files(r, folder):
    """Every shot's files: the checked ones StoryStudio already knows, plus any new ones in
    the folder (those are measured here, and used whole)."""
    try:
        found, _, _ = K.find_files(r, folder) if folder else ({}, {}, [])
    except StoryError:
        found = {}
    out = {}
    for s in r["shots"]:
        known = {f["path"]: f for f in s.get("files") or []}
        here = list(found.get(s["n"], []))
        files = []
        for path in here + [p for p in known if p not in here]:
            p = Path(path)
            if not p.exists():
                continue
            mtime = int(p.stat().st_mtime)
            f = known.get(path)
            if f and f.get("mtime") != mtime:
                f = None                                  # changed since it was checked
            if not f:
                kind = "clip" if p.suffix.lower() in K.CLIPS else "picture"
                f = {"path": path, "name": p.name, "kind": kind, "mtime": mtime}
            if f["kind"] == "clip" and not f.get("seconds"):
                try:
                    info = probe(path)
                    f = dict(f, seconds=round(info.get("duration") or 0, 2),
                             fps=round(info.get("fps") or 0, 2),
                             w=info.get("width") or 0, h=info.get("height") or 0)
                except Exception:
                    log_error("rough cut: measuring " + p.name)
                    continue
            files.append(f)
        out[s["n"]] = files
    return out


def plan(r, folder=None):
    """What goes on the timeline, in order: (items, missing). An item is a dict with the
    file, the seconds of it to use, the speed, and whether it is only a stand-in."""
    files = _files(r, folder or r.get("clip_folder") or "")
    for s in r["shots"]:                 # new files are remembered, like the clip check does
        s["files"] = files.get(s["n"], [])
        K._status(r, s)
    items, missing = [], []
    for s in sorted(r["shots"], key=lambda s: s["n"]):
        mine = files.get(s["n"], [])
        clips = [f for f in mine if f["kind"] == "clip" and (f.get("seconds") or 0) > 0.3]
        pics = [f for f in mine if f["kind"] == "picture"]
        good = sorted((f for f in clips if _usable(f)),
                      key=lambda f: (K.RANK.get(f["check"]["verdict"], 3), -f["mtime"]))
        fresh = sorted((f for f in clips if not f.get("check")), key=lambda f: -f["mtime"])
        want = float(s.get("end") or 0) - float(s.get("start") or 0)
        want = want if 1.0 <= want <= 20.0 else HOLD
        it = {"n": s["n"], "title": s.get("title", ""), "speed": 100, "stand_in": False,
              "checked": True, "extra": bool(s.get("extra"))}
        if good:
            f, c = good[0], good[0]["check"]
            lo = float(c.get("best_from") or 0)
            hi = float(c.get("best_to") or 0) or f["seconds"]
            lo = max(0.0, min(lo, f["seconds"] - 0.2))
            hi = max(lo + 0.2, min(hi, f["seconds"]))
            speed = int(c.get("speed") or 100)
            it.update(file=f, a=lo, b=hi,
                      speed=max(50, min(300, speed)) if abs(speed - 100) >= 5 else 100)
        elif fresh:
            f = fresh[0]
            it.update(file=f, a=0.0, b=f["seconds"], checked=False)
        elif pics:
            f = max(pics, key=lambda f: ((f.get("check") or {}).get("verdict") == "keep",
                                         f["mtime"]))
            it.update(file=f, a=0.0, b=want, stand_in=True)
        elif clips:
            f = max(clips, key=lambda f: f["mtime"])
            it.update(file=f, a=0.0, b=f["seconds"], stand_in=True)
        else:
            missing.append(s["n"])
            it = {"n": s["n"], "title": s.get("title", ""), "file": None}
        items.append(it)
    return items, missing


# ---------------------------------------------------------------------------
# the file Premiere imports
# ---------------------------------------------------------------------------
def _url(path):
    return "file://localhost/" + quote(Path(path).resolve().as_posix(), safe="/")


def _rate_xml(tb, ntsc):
    return f"<rate><timebase>{tb}</timebase><ntsc>{'TRUE' if ntsc else 'FALSE'}</ntsc></rate>"


def _param(pid, name, value, lo=None, hi=None):
    rng = f"<valuemin>{lo}</valuemin><valuemax>{hi}</valuemax>" if lo is not None else ""
    return (f'<parameter authoringApp="PremierePro"><parameterid>{pid}</parameterid>'
            f"<name>{name}</name>{rng}<value>{value}</value></parameter>")


def _speed_xml(speed):
    return ("<filter><effect><name>Time Remap</name><effectid>timeremap</effectid>"
            "<effectcategory>motion</effectcategory><effecttype>motion</effecttype>"
            "<mediatype>video</mediatype>"
            + _param("variablespeed", "variablespeed", 0, 0, 1)
            + _param("speed", "speed", speed, -100000, 100000)
            + _param("reverse", "reverse", "FALSE")
            + _param("frameblending", "frameblending", "FALSE")
            + "</effect></filter>")


def _scale_xml(scale):
    return ("<filter><effect><name>Basic Motion</name><effectid>basic</effectid>"
            "<effectcategory>motion</effectcategory><effecttype>motion</effecttype>"
            "<mediatype>video</mediatype>"
            + _param("scale", "Scale", f"{scale:.1f}", 0, 1000)
            + "</effect></filter>")


def _picture_size(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        return 0, 0


def _frame_size(r, items):
    sizes = {}
    for it in items:
        f = it.get("file")
        if f and f["kind"] == "clip" and f.get("w") and f.get("h"):
            sizes[(f["w"], f["h"])] = sizes.get((f["w"], f["h"]), 0) + 1
    if sizes:
        return max(sizes, key=sizes.get)
    try:
        a, b = (float(x) for x in str(r.get("aspect") or "16:9").split(":"))
        return (1920, int(round(1920 * b / a / 2) * 2)) if a >= b else \
            (int(round(1920 * a / b / 2) * 2), 1920)
    except (ValueError, ZeroDivisionError):
        return 1920, 1080


def xml(r, items, name):
    """The timeline as text, plus what was placed: (text, placed, seconds)."""
    rates = {}
    for it in items:
        f = it.get("file")
        if f and f["kind"] == "clip" and f.get("fps"):
            k = _rate(f["fps"])
            rates[k] = rates.get(k, 0) + 1
    tb, ntsc = max(rates, key=rates.get) if rates else (24, False)
    fps = _real(tb, ntsc)
    W, H = _frame_size(r, items)
    rate = _rate_xml(tb, ntsc)

    def sample(w, h):
        return (f"<samplecharacteristics>{rate}<width>{w}</width><height>{h}</height>"
                "<anamorphic>FALSE</anamorphic><pixelaspectratio>square</pixelaspectratio>"
                "<fielddominance>none</fielddominance></samplecharacteristics>")
    clips, markers, placed, cursor = [], [], [], 0
    for k, it in enumerate(items, 1):
        f = it.get("file")
        if not f:
            markers.append((cursor, f"Shot {it['n']} goes here",
                            "Not made yet: " + it.get("title", "")))
            continue
        ratio = it["speed"] / 100.0
        length = max(1, int(round((it["b"] - it["a"]) / ratio * fps)))
        if f["kind"] == "clip":
            w, h = f.get("w") or W, f.get("h") or H
            ftb, fntsc = _rate(f.get("fps") or fps)
            file_frames = max(1, int(f["seconds"] * _real(ftb, fntsc)))
            total = max(1, int(f["seconds"] * fps / ratio))      # at the speed it plays
            length = min(length, total)
            start_in = max(0, min(int(round(it["a"] / ratio * fps)), total - length))
            media = (_rate_xml(ftb, fntsc) + f"<duration>{file_frames}</duration>"
                     f"<media><video><duration>{file_frames}</duration>"
                     + sample(w, h) + "</video></media>")
        else:
            w, h = _picture_size(f["path"])
            w, h = w or W, h or H
            total, start_in = STILL_FRAMES, STILL_IN
            media = (f"{rate}<media><video><duration>{STILL_FRAMES}</duration>"
                     "<stillframe>TRUE</stillframe>" + sample(w, h) + "</video></media>")
        filters = _speed_xml(it["speed"]) if it["speed"] != 100 else ""
        scale = min(W / w, H / h) * 100
        if abs(scale - 100) > 1:
            filters += _scale_xml(scale)
        colour = STAND_IN_COLOUR if it["stand_in"] else \
            (UNCHECKED_COLOUR if not it["checked"] else "")
        clips.append(
            f'<clipitem id="clipitem-{k}"><masterclipid>masterclip-{k}</masterclipid>'
            f"<name>{escape(f['name'])}</name><enabled>TRUE</enabled>"
            f"<duration>{total}</duration>{rate}"
            f"<start>{cursor}</start><end>{cursor + length}</end>"
            f"<in>{start_in}</in><out>{start_in + length}</out>"
            "<alphatype>none</alphatype><pixelaspectratio>square</pixelaspectratio>"
            "<anamorphic>FALSE</anamorphic>"
            f'<file id="file-{k}"><name>{escape(f["name"])}</name>'
            f"<pathurl>{escape(_url(f['path']))}</pathurl>{media}</file>"
            + filters
            + (f"<labels><label2>{colour}</label2></labels>" if colour else "")
            + "</clipitem>")
        if it["stand_in"]:
            markers.append((cursor, f"Shot {it['n']}: stand-in",
                            "This shot's clip isn't ready yet, so "
                            + ("its picture" if f["kind"] == "picture" else "the last try")
                            + " is holding its place."))
        placed.append(dict(it, at=round(cursor / fps, 2), length=round(length / fps, 2)))
        cursor += length
    if not clips:
        raise StoryError("There are no pictures or clips to put on a timeline yet.")
    marks = "".join(f"<marker><comment>{escape(c)}</comment><name>{escape(n)}</name>"
                    f"<in>{max(0, min(t, cursor - 1))}</in><out>-1</out></marker>"
                    for t, n, c in markers)
    text = ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="4">'
            f'<sequence id="sequence-1"><name>{escape(name)}</name>'
            f"<duration>{cursor}</duration>{rate}"
            f"<timecode>{rate}<string>00:00:00:00</string><frame>0</frame>"
            "<displayformat>NDF</displayformat></timecode>"
            "<media><video><format>" + sample(W, H) + "</format>"
            "<track>" + "".join(clips) + "<enabled>TRUE</enabled><locked>FALSE</locked></track>"
            "</video><audio><numOutputChannels>2</numOutputChannels><format>"
            "<samplecharacteristics><depth>16</depth><samplerate>48000</samplerate>"
            "</samplecharacteristics></format></audio></media>"
            + marks + "</sequence></xmeml>\n")
    return text, placed, cursor / fps


def build(r, folder=None):
    """Write the rough cut next to the pictures and clips, and say what went into it."""
    import ss_script as S
    folder = folder or r.get("clip_folder") or ""
    if not folder or not Path(folder).is_dir():
        raise StoryError("Choose the folder with your pictures and clips first.")
    items, missing = plan(r, folder)
    title = (r.get("title") or "My video").strip()
    name = f"{title} rough cut"
    text, placed, seconds = xml(r, items, name)
    out = Path(folder) / f"{safe_name(title)} - rough cut.xml"
    try:
        out.write_text(text, encoding="utf8")
    except OSError:
        raise StoryError("StoryStudio couldn't save the rough cut in that folder. Is the "
                         "folder still there, and not read-only?")
    res = {
        "file": str(out), "file_name": out.name, "name": name, "folder": str(folder),
        "shots": len(placed), "seconds": round(seconds, 1),
        "ready": sum(1 for p in placed if not p["stand_in"]),
        "trimmed": sum(1 for p in placed if p["file"]["kind"] == "clip" and not p["stand_in"]
                       and (p["a"] > 0.15 or p["b"] < p["file"]["seconds"] - 0.15)),
        "sped": sum(1 for p in placed if p["speed"] != 100),
        "unchecked": [p["n"] for p in placed if not p["checked"]],
        "stand_ins": [p["n"] for p in placed if p["stand_in"]],
        "missing": missing, "made": time.strftime("%Y-%m-%d %H:%M"),
    }
    r["clip_folder"] = str(folder)
    r["rough_cut"] = res
    try:
        r["files"] = S.save(r)
    except Exception:
        log_error("rough cut: saving the shot list")
    log("rough cut:", out.name, res["shots"], "shots,", res["seconds"], "s; stand-ins",
        res["stand_ins"], "missing", missing)
    return res
