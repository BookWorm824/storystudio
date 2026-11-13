"""Read Adobe Premiere Pro projects (.prproj) without Premiere.

A .prproj is gzip-compressed XML. Every object sits directly under <PremiereData> with an
ObjectID (referenced by ObjectRef) or ObjectUID (referenced by ObjectURef). The parts we use:

  Sequence -> TrackGroups -> VideoTrackGroup / AudioTrackGroup -> Tracks
  VideoClipTrack / AudioClipTrack -> ClipItems -> VideoClipTrackItem / AudioClipTrackItem
  TrackItem (Start, End: timeline ticks) -> SubClip -> VideoClip / AudioClip (InPoint, OutPoint)
  Clip -> Source (VideoMediaSource / AudioMediaSource / *SequenceSource) -> Media (file paths)

Times are in ticks: 254016000000 per second. Clip volume lives in an AudioComponentChain as
an "Internal Volume ..." filter whose "Level" param is linear gain * 10^(-15/20).
"""
import gzip
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from ss_core import TICKS, StoryError, log, log_error

MEDIA_EXTS = {".mp4", ".mov", ".m4v", ".mxf", ".avi", ".mkv", ".webm", ".wav", ".mp3", ".aif",
              ".aiff", ".m4a", ".aac", ".flac", ".ogg", ".png", ".jpg", ".jpeg", ".psd",
              ".tif", ".tiff", ".gif", ".bmp", ".webp"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".psd", ".tif", ".tiff", ".gif", ".bmp", ".webp"}


def level_to_db(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if v <= 0:
        return -96.0
    return 20 * math.log10(v) + 15.0


def db_to_level(db):
    return 10 ** ((db - 15.0) / 20.0)


def read_xml(path):
    raw = Path(path).read_bytes()
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass                      # some tools save it uncompressed
    return raw


class Project:
    def __init__(self, path, home=None):
        """home: where the project really lives, when `path` is StoryStudio's own saved copy
        of it (moved media are looked for around the real project, not the copy)."""
        self.path = Path(path)
        self.home = Path(home) if home else self.path
        try:
            self.root = ET.fromstring(read_xml(self.path))
        except (OSError, EOFError, ET.ParseError):
            log_error("Project.load")
            raise StoryError("StoryStudio couldn't open that Premiere project. If Premiere has it "
                             "open, save it and try again.")
        self.objs = {}
        for el in self.root:
            k = el.get("ObjectID") or el.get("ObjectUID")
            if k:
                self.objs[k] = el
        self._file_index = None

    # --- object helpers ----------------------------------------------------
    def ref(self, el, path):
        """Follow a child element's ObjectRef / ObjectURef."""
        if el is None:
            return None
        c = el.find(path)
        if c is None:
            return None
        k = c.get("ObjectRef") or c.get("ObjectURef")
        return self.objs.get(k)

    def refs(self, el, path):
        out = []
        if el is None:
            return out
        for c in el.findall(path):
            k = c.get("ObjectRef") or c.get("ObjectURef")
            if k in self.objs:
                out.append(self.objs[k])
        return out

    @staticmethod
    def num(el, path, default=0):
        if el is None:
            return default
        t = el.findtext(path)
        try:
            return int(t) if t is not None else default
        except ValueError:
            try:
                return float(t)
            except ValueError:
                return default

    # --- sequences ---------------------------------------------------------
    def sequences(self):
        out = []
        for el in self.root:
            if el.tag == "Sequence":
                seq = self.read_sequence(el)
                if seq and seq["duration"] > 0:
                    out.append(seq)
        out.sort(key=lambda s: -s["duration"])
        return out

    def main_sequence(self):
        seqs = self.sequences()
        if not seqs:
            raise StoryError("That project has no sequence with clips in it.")
        stem = self.home.stem.lower()
        for s in seqs:                      # one named like the project wins
            if s["name"].lower().strip() == stem:
                return s
        return seqs[0]

    def read_sequence(self, seq_el, depth=0):
        name = seq_el.findtext("Name") or "Sequence"
        fps, width, height = 24.0, 1920, 1080
        video, audio = [], []
        for g in seq_el.findall("TrackGroups/TrackGroup"):
            grp = self.ref(g, "Second")
            if grp is None:
                continue
            if grp.tag == "VideoTrackGroup":
                fr = self.num(grp, "TrackGroup/FrameRate", 0)
                if fr:
                    fps = TICKS / fr
                rect = (grp.findtext("FrameRect") or "").split(",")
                if len(rect) == 4:
                    try:
                        width, height = int(rect[2]), int(rect[3])
                    except ValueError:
                        pass
                for ti, tr in enumerate(self.refs(grp, "TrackGroup/Tracks/Track")):
                    video += self._track_items(tr, "video", ti, depth)
            elif grp.tag == "AudioTrackGroup":
                for ti, tr in enumerate(self.refs(grp, "TrackGroup/Tracks/Track")):
                    chain = self.ref(tr, "AudioTrack/ComponentOwner/Components")
                    if self._track_muted(chain):
                        continue                # a muted track is not part of the movie's sound
                    track_db = self._volume_db(chain)
                    for it in self._track_items(tr, "audio", ti, depth):
                        it["gain_db"] += track_db
                        it["track_db"] = track_db
                        audio.append(it)
        dur = max([i["end"] for i in video + audio] or [0.0])
        return {"name": name, "uid": seq_el.get("ObjectUID") or seq_el.get("ObjectID"),
                "fps": fps, "width": width, "height": height, "duration": dur,
                "video": sorted(video, key=lambda i: (i["start"], -i["track"])),
                "audio": sorted(audio, key=lambda i: (i["start"], i["track"]))}

    def _track_items(self, track_el, kind, track_index, depth):
        items = []
        ct = track_el.find("ClipTrack")
        if ct is None:
            return items
        muted = False
        for it_el in self.refs(ct, "ClipItems/TrackItems/TrackItem"):
            cti = it_el.find("ClipTrackItem")
            if cti is None:
                continue
            start = self.num(cti, "TrackItem/Start", 0) / TICKS
            end = self.num(cti, "TrackItem/End", 0) / TICKS
            if end <= start:
                continue
            sub = self.ref(cti, "SubClip")
            clip = self.ref(sub, "Clip")
            if clip is None:
                continue
            c = clip.find("Clip")
            src_in = self.num(c, "InPoint", 0) / TICKS
            src_out = self.num(c, "OutPoint", 0) / TICKS
            speed = 1.0
            if src_out > src_in:
                speed = (src_out - src_in) / (end - start)
            source = self.ref(c, "Source")
            tag = ""
            props = cti.find("TrackItem/Node/Properties")
            if props is not None:
                tag = (props.findtext("ESP.Tag") or props.findtext("ESP.AdjustmentsModeID")
                       or "").lower()
            item = {"kind": kind, "track": track_index, "start": start, "end": end,
                    "src_in": src_in, "speed": speed if 0.05 < speed < 20 else 1.0,
                    "name": (sub.findtext("Name") if sub is not None else "") or "",
                    "tag": tag, "gain_db": 0.0, "file": "", "nested": None, "muted": muted,
                    "id": it_el.get("ObjectID") or it_el.get("ObjectUID")}
            if kind == "audio":
                item["gain_db"] = self._volume_db(self.ref(cti, "ComponentOwner/Components"))
                try:                             # the clip's Audio Gain (a plain multiplier)
                    g = float(clip.findtext("Gain") or 1.0)
                    if g > 0 and abs(g - 1.0) > 1e-4:
                        item["gain_db"] += 20 * math.log10(g)
                except ValueError:
                    pass
            if source is not None and "SequenceSource" in source.tag:
                nested = self.ref(source, "SequenceSource/Sequence")
                if nested is not None and depth < 4:
                    item["nested"] = self.read_sequence(nested, depth + 1)
            elif source is not None:
                media = self.ref(source, "MediaSource/Media")
                item["file"] = self.find_media(media)
                item["original_path"] = ""
                if media is not None:
                    item["original_path"] = (media.findtext("ActualMediaFilePath") or
                                             media.findtext("FilePath") or
                                             media.findtext("Title") or "")
            if item["file"] or item["nested"]:
                items.append(item)
            elif source is not None and "SequenceSource" not in source.tag:
                items.append(item)          # missing media: kept so we can report it
        return items

    def _track_muted(self, chain):
        """The track's mute button (the "Mute" setting on its fader) is on."""
        if chain is None:
            return False
        for comp in self.refs(chain, "ComponentChain/Components/Component"):
            for p in self.refs(comp, "AudioComponent/Component/Params/Param"):
                if p.findtext("Name") == "Mute":
                    v = (p.findtext("CurrentValue") or "").strip().lower()
                    if v in ("true", "false", ""):
                        return v == "true"
                    try:
                        return float(v) > 0.5
                    except ValueError:
                        return False
        return False

    def _volume_db(self, chain):
        """Gain in dB from an AudioComponentChain's volume filter (0 if unchanged)."""
        if chain is None or chain.findtext("DefaultVol") == "true":
            return 0.0
        for comp in self.refs(chain, "ComponentChain/Components/Component"):
            if "Volume" not in (comp.findtext("FilterMatchName") or ""):
                continue
            for p in self.refs(comp, "AudioComponent/Component/Params/Param"):
                if p.findtext("Name") == "Level":
                    return level_to_db(p.findtext("CurrentValue"))
        return 0.0

    # --- finding media that moved ------------------------------------------
    def find_media(self, media):
        if media is None:
            return ""
        cands = [media.findtext("ActualMediaFilePath"), media.findtext("FilePath")]
        rel = media.findtext("RelativePath")
        if rel:
            cands.append(str((self.home.parent / rel.replace("\\", os.sep))))
        for i in range(5):
            cands.append(media.findtext(f"MediaFileHistory{i}"))
        for c in cands:
            if c and os.path.isfile(c):
                return os.path.normpath(c)
        title = media.findtext("Title") or ""
        name = Path((cands[0] or cands[1] or title).replace("\\", "/")).name or title
        if not name:
            return ""
        hits = self.file_index().get(name.lower(), [])
        if not hits:
            return ""
        # many movies have their own "Video Shots\Shot 1.mp4": prefer the file whose folders
        # match the original path the furthest back, then one inside this movie's own folder
        orig = [x.lower() for x in Path((cands[0] or cands[1] or "").replace("\\", "/")).parts]
        home = self.home.parent
        if home.name.lower() == "project files":
            home = home.parent
        home = str(home).lower()

        def score(p):
            parts = [x.lower() for x in Path(p).parts]
            n = 0
            while n < min(len(parts), len(orig)) and parts[-1 - n] == orig[-1 - n]:
                n += 1
            return (-n, not p.lower().startswith(home), len(p))
        hits = sorted(hits, key=score)
        return hits[0]

    def file_index(self):
        if self._file_index is None:
            idx = {}
            roots = [self.home.parent, self.home.parent.parent]
            if self.home.parent.name.lower() == "project files":
                roots.append(self.home.parent.parent.parent)
            seen = 0
            for root in roots:
                for dirpath, dirnames, filenames in os.walk(root):
                    dirnames[:] = [d for d in dirnames if not d.startswith(".") and
                                   d.lower() not in ("adobe premiere pro auto-save",
                                                     "adobe premiere pro video previews",
                                                     "adobe premiere pro audio previews")]
                    for f in filenames:
                        if Path(f).suffix.lower() in MEDIA_EXTS:
                            idx.setdefault(f.lower(), [])
                            p = os.path.join(dirpath, f)
                            if p not in idx[f.lower()]:
                                idx[f.lower()].append(p)
                    seen += len(filenames)
                    if seen > 60000:
                        break
            self._file_index = idx
        return self._file_index


# ---------------------------------------------------------------------------
# flattening a sequence into what plays when
# ---------------------------------------------------------------------------
def flatten(seq, offset=0.0, t0=None, t1=None):
    """Video and audio items with nested sequences expanded, times on the main timeline."""
    video, audio = [], []
    lo = 0.0 if t0 is None else t0
    hi = seq["duration"] if t1 is None else t1
    for kind, items, out in (("video", seq["video"], video), ("audio", seq["audio"], audio)):
        for it in items:
            s, e = max(it["start"], lo), min(it["end"], hi)
            if e <= s:
                continue
            if it.get("nested"):
                # nested time = src_in + (t - start) * speed (speed assumed 1 for nests)
                ns = it["src_in"] + (s - it["start"])
                ne = it["src_in"] + (e - it["start"])
                v, a = flatten(it["nested"], offset + it["start"] - it["src_in"], ns, ne)
                for x in (v if kind == "video" else a):
                    x = dict(x)
                    x["track"] = it["track"] + x["track"] / 100.0
                    if kind == "audio":
                        x["gain_db"] += it["gain_db"]
                    out.append(x)
                continue
            x = dict(it)
            x["src_in"] = it["src_in"] + (s - it["start"]) * it["speed"]
            x["start"], x["end"] = s + offset, e + offset
            out.append(x)
    return video, audio


def picture_segments(seq):
    """Non-overlapping (start, end, item) of the top-most video clip, with gaps as None."""
    video, _ = flatten(seq)
    video = [v for v in video if v.get("file")]
    cuts = sorted({0.0, seq["duration"]} | {v["start"] for v in video} | {v["end"] for v in video})
    segs = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a < 1e-4:
            continue
        mid = (a + b) / 2
        top = None
        for v in video:
            if v["start"] <= mid < v["end"] and (top is None or v["track"] > top["track"]):
                top = v
        if segs and segs[-1][2] is top:
            segs[-1] = (segs[-1][0], b, top)
        else:
            segs.append((a, b, top))
    return segs


def shot_changes(seq):
    """Times where the picture changes to a different clip (exact, from the edit)."""
    out = []
    for a, b, it in picture_segments(seq)[1:]:
        out.append(round(a, 3))
    return out


def summary(seq):
    v, a = flatten(seq)
    missing = sorted({Path(x.get("original_path") or x["name"]).name
                      for x in v + a if not x.get("file")})
    return {"name": seq["name"], "duration": seq["duration"], "fps": seq["fps"],
            "shots": len([s for s in picture_segments(seq) if s[2] is not None]),
            "audio_clips": len(a), "missing": missing,
            "dialogue_clips": len([x for x in a if x["tag"].startswith("dialog")])}


def find_export(project_path, seq):
    """An exported movie of this sequence (same name, same length), if there is one."""
    from ss_core import probe
    p = Path(project_path)
    names = {seq["name"].lower().strip(), p.stem.lower().strip()}
    roots = [p.parent, p.parent.parent, p.parent.parent.parent]
    found = []
    for root in roots:
        for sub in [root, root / "Finished Video", root / "Exports", root / "Export"]:
            if not sub.is_dir():
                continue
            for f in sub.iterdir():
                if f.suffix.lower() in (".mp4", ".mov", ".m4v") and \
                        f.stem.lower().strip() in names:
                    found.append(f)
    for f in found:
        try:
            if abs(probe(str(f))["duration"] - seq["duration"]) < 1.5:
                return str(f)
        except Exception:
            log_error("find_export probe")
    return ""
