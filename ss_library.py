"""Real recorded sound effects from Adobe's free sound effects library.

Adobe gives away about 10,000 professionally recorded sound effects for its Audition app, in
26 zip packs on download.adobe.com, royalty-free for use in productions (Adobe's licence:
the sound files can't be handed out on their own). No account is needed.

The packs are big (up to 1.4 GB), so StoryStudio never downloads a whole one. A zip keeps a
table of contents at its end: StoryStudio reads just that (a few kilobytes) to learn every
file's name and where it sits, then fetches only the sounds a story needs, one at a time,
and keeps them for next time.
"""
import json
import re
import struct
import threading
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ss_core import APP_DATA, log

BASE = "https://download.adobe.com/pub/adobe/audition/offers/adobeauditiondlcsfx/"
PACKS = ["ambience_1", "ambience_2", "animals", "cartoon", "crashes", "drones",
         "emergency_effects", "fire_explosions", "foley", "foley_footsteps", "horror",
         "human_elements", "imaging_elements", "impacts", "industry", "liquid_water",
         "multimedia", "noise_tones_dtmf_tests", "production_elements", "science_fiction",
         "sports", "technology", "transportation", "underwater", "weapons", "weather"]
# packs never used for a family storybook (alarms, guns, explosions, test tones, horror)
SKIP = {"emergency_effects", "fire_explosions", "weapons", "noise_tones_dtmf_tests", "horror",
        "drones", "science_fiction", "imaging_elements"}
HOME = APP_DATA / "library"
INDEX = HOME / "adobe_index.json"
SOUNDS = HOME / "adobe"
CREDIT = "Adobe Audition sound effects (royalty-free)"
_lock = threading.Lock()
_index = None


def _get(url, start=None, end=None, timeout=60):
    h = {"User-Agent": "StoryStudio"}
    if start is not None:
        h["Range"] = f"bytes={start}-{end}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
        return r.read()


def _size(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "StoryStudio"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def _pack_files(pack):
    """Every sound in one pack: [{name, pack, at, csize, size, method}] from its table of
    contents (handles the large-zip "zip64" form too)."""
    url = BASE + pack + ".zip"
    size = _size(url)
    tail = _get(url, max(0, size - 70000), size - 1)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise ValueError("no table of contents")
    n, cd_size, cd_off = struct.unpack("<HII", tail[i + 10:i + 20])
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:          # zip64
        j = tail.rfind(b"PK\x06\x06")
        cd_size, cd_off = struct.unpack("<QQ", tail[j + 40:j + 56])
    cd = _get(url, cd_off, cd_off + cd_size - 1, timeout=120)
    out, p = [], 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", cd[p + 10:p + 12])[0]
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        fl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
        at = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + fl].decode("utf8", "replace")
        extra = cd[p + 46 + fl:p + 46 + fl + el]
        if 0xFFFFFFFF in (csize, usize, at):                   # zip64 sizes in the extra field
            q = 0
            while q + 4 <= len(extra):
                tag, ln = struct.unpack("<HH", extra[q:q + 4])
                if tag == 1:
                    vals, k = [], q + 4
                    for want in (usize, csize, at):
                        if want == 0xFFFFFFFF:
                            vals.append(struct.unpack("<Q", extra[k:k + 8])[0])
                            k += 8
                        else:
                            vals.append(want)
                    usize, csize, at = vals
                q += 4 + ln
        base = name.replace("\\", "/").split("/")[-1]
        if base.lower().endswith((".wav", ".aif", ".aiff", ".mp3")) and not base.startswith("._"):
            out.append({"name": base, "pack": pack, "at": at, "csize": csize, "size": usize,
                        "method": method})
        p += 46 + fl + el + cl
    return out


def index(refresh=False):
    """Every sound in the library (read once, then kept on this computer)."""
    global _index
    with _lock:
        if _index is not None and not refresh:
            return _index
        if INDEX.exists() and not refresh:
            try:
                _index = json.loads(INDEX.read_text(encoding="utf8"))
                return _index
            except ValueError:
                pass
        HOME.mkdir(parents=True, exist_ok=True)
        found = []

        def one(pack):
            try:
                return _pack_files(pack)
            except Exception as ex:
                log("sound library: couldn't read", pack, type(ex).__name__, ex)
                return []
        with ThreadPoolExecutor(6) as ex:
            for files in ex.map(one, [p for p in PACKS if p not in SKIP]):
                found += files
        if found:
            INDEX.write_text(json.dumps(found), encoding="utf8")
        _index = found
        log("sound library: indexed", len(found), "sounds")
        return found


def title(entry):
    """The plain words of a sound's name: "Foley Trunk Latch Opening 01.wav" ->
    "Foley Trunk Latch Opening"."""
    t = re.sub(r"\.(wav|aiff?|mp3)$", "", entry["name"], flags=re.I)
    return re.sub(r"\s+\d+$", "", t).strip()


WORD = re.compile(r"[a-z]+")
SYN = {"creak": ["creak", "creaking", "squeak"], "steps": ["footstep", "footsteps", "walking",
       "step"], "footsteps": ["footstep", "walking", "step"], "writing": ["write", "pen",
       "pencil", "scribble"], "pencil": ["pencil", "pen", "write", "scribble"],
       "laugh": ["laugh", "laughing", "giggle", "chuckle"], "giggle": ["giggle", "laugh"],
       "chuckle": ["chuckle", "laugh"], "birds": ["bird", "birds", "songbird"],
       "bird": ["bird", "birds"], "trunk": ["trunk", "chest", "latch", "lid"],
       "chest": ["chest", "trunk", "lid"], "kids": ["children", "child", "kids"],
       "children": ["children", "child", "kids"], "dishes": ["dishes", "plate", "cutlery"],
       "diner": ["restaurant", "cafe", "diner"], "restaurant": ["restaurant", "cafe"],
       "chitter": ["chatter", "squeak", "chirp"], "chitters": ["chatter", "squeak", "chirp"],
       "chittering": ["chatter", "squeak", "chirp"], "squeal": ["squeal", "squeak"],
       "squeaky": ["squeak", "squeaky"], "huff": ["breath", "exhale", "sigh"],
       "whimper": ["whimper", "whine"], "panting": ["pant", "panting", "breathing"],
       "pant": ["pant", "panting", "breathing"], "chuckle": ["chuckle", "laugh", "giggle"],
       "strain": ["strain", "grunt", "effort"], "straining": ["strain", "grunt", "effort"],
       "effort": ["effort", "grunt", "strain"], "gulp": ["gulp", "swallow"],
       "yelp": ["yelp", "scream", "shriek"], "surprise": ["gasp", "surprise"],
       "zip": ["zip", "whizz", "whoosh"], "whizz": ["whizz", "whoosh", "zip"],
       "dash": ["whoosh", "zip", "run"], "thud": ["thud", "thump", "hit"],
       "plunk": ["plunk", "drop", "pop"], "boing": ["boing", "spring"],
       "xylophone": ["xylophone", "glockenspiel", "marimba"], "ding": ["ding", "bell"],
       "rumble": ["rumble", "rolling", "roll"], "scamper": ["footstep", "run", "scurry"],
       "patter": ["footstep", "patter", "run"], "rustle": ["rustle", "leaves"],
       "creek": ["creek", "stream", "river"], "stream": ["stream", "creek", "river"],
       "stretch": ["stretch", "rubber", "squeak"], "tok": ["knock", "wood", "tap"]}
STOP = {"a", "an", "the", "and", "of", "on", "in", "with", "to", "as", "at", "her", "his",
        "it", "is", "soft", "gentle", "softly", "light", "slowly", "slow", "small", "quiet",
        "warm", "faint", "short", "long", "close", "up", "down", "into", "from", "by",
        "wordless", "voice", "no", "words", "speech", "never", "cute", "tiny", "little",
        "quick", "about", "seconds", "second", "one", "two", "three", "like", "high",
        "pitched", "playful", "then", "each", "very", "bit", "sound", "sounds", "comic"}


def candidates(text, n=30, long=False):
    """The library sounds whose names best match these words (most likely first)."""
    words = [w for w in WORD.findall((text or "").lower()) if w not in STOP]
    want = set()
    for w in words:
        want.update(SYN.get(w, [w]))
        if w.endswith("s") and len(w) > 3:
            want.add(w[:-1])
    seen, scored = set(), []
    for e in index():
        t = title(e)
        if t.lower() in seen:
            continue
        name = set(WORD.findall(t.lower()))
        hit = len(want & name)
        if not hit:
            continue
        amb = e["pack"].startswith("ambience") or "ambience" in name
        score = hit + (0.6 if long == amb else 0)
        scored.append((score, -e["size"] if long else 0, t, e))
        seen.add(t.lower())
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [e for *_, e in scored[:n]]


def fetch(entry):
    """The sound file on this computer (fetched the first time it's needed). Returns its
    path, or None if it couldn't be fetched."""
    SOUNDS.mkdir(parents=True, exist_ok=True)
    dest = SOUNDS / entry["pack"] / entry["name"]
    if dest.exists() and dest.stat().st_size == entry["size"]:
        return str(dest)
    url = BASE + entry["pack"] + ".zip"
    try:
        head = _get(url, entry["at"], entry["at"] + 29)
        if head[:4] != b"PK\x03\x04":
            raise ValueError("not where the table of contents said")
        fl, el = struct.unpack("<HH", head[26:30])
        start = entry["at"] + 30 + fl + el
        raw = _get(url, start, start + entry["csize"] - 1, timeout=300)
        if entry["method"] == 8:
            raw = zlib.decompress(raw, -15)
        elif entry["method"] != 0:
            raise ValueError(f"packed in a way StoryStudio can't read ({entry['method']})")
        if len(raw) != entry["size"]:
            raise ValueError("came back the wrong size")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(dest) + ".part")
        tmp.write_bytes(raw)
        tmp.replace(dest)
        return str(dest)
    except Exception as ex:
        log("sound library: couldn't fetch", entry["name"], type(ex).__name__, ex)
        return None


def fetch_many(entries, progress=None):
    """{name: path} for every entry that could be fetched (a few at a time)."""
    out, done = {}, [0]
    lock = threading.Lock()

    def one(e):
        p = fetch(e)
        with lock:
            done[0] += 1
            if p:
                out[e["name"]] = p
            if progress:
                progress(done[0] / max(1, len(entries)),
                         f"Getting real recorded sounds: {done[0]} of {len(entries)}")
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, entries))
    log("sound library: fetched", len(out), "of", len(entries), f"in {time.time() - t0:.0f}s")
    return out
