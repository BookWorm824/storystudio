"""Write sound effects into a COPY of a Premiere project (.prproj). The original is never
changed.

How: a new audio clip is made by cloning the object graph of an existing audio-only clip in
the same project (AudioClipTrackItem -> SubClip -> AudioClip -> AudioMediaSource -> Media ->
AudioStream, plus MasterClip, ClipLoggingInfo, channel serializers, markers, component chains)
with fresh IDs, then pointing it at the new file and setting its length and timeline position.
A new audio track is a clone of an existing AudioClipTrack (with its fader/meter/panner),
added to the sequence's AudioTrackGroup and to the mixer's AudioTrackInlet.
Premiere rebuilds caches (peak files, conformed audio) for the new media when it opens.
"""
import copy
import gzip
import os
import re
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from ss_core import TICKS, StoryError, log, log_error
import ss_premiere as P

CLIP_TAGS = {"AudioClipTrackItem", "SubClip", "MasterClip", "AudioClip", "AudioMediaSource",
             "Media", "AudioStream", "SecondaryContent", "ClipChannelGroupVectorSerializer",
             "ClipChannelVectorSerializer", "ClipChannelSerializer", "ClipLoggingInfo",
             "Markers", "AudioComponentChain", "AudioFilterComponent", "AudioComponentParam",
             "ClipProjectItem"}
TRACK_TAGS = {"AudioClipTrack", "AudioComponentChain", "AudioFilterComponent",
              "AudioComponentParam", "AudioMeter", "AudioFader", "StereoToStereoPanProcessor",
              "DefaultPanProcessor", "StereoPanProcessor", "MonoToStereoPanProcessor"}
AUDIO_ONLY = {".wav", ".mp3", ".aif", ".aiff", ".m4a", ".aac", ".flac", ".ogg"}


def _plain_speed(clip):
    """A copied clip plays at normal speed, forwards: the shot it was copied from may have
    been sped up, slowed down or reversed in Premiere (no PlaybackSpeed means 100%)."""
    if clip is None:
        return
    for x in list(clip):
        if x.tag == "PlaybackSpeed" or "backward" in x.tag.lower() or                 "reverse" in x.tag.lower():
            clip.remove(x)


def _speed_of(w, track_item):
    try:
        sub = w.get(track_item, "ClipTrackItem/SubClip")
        clip = w.get(sub, "Clip") if sub is not None else None
        return abs(float(clip.findtext("Clip/PlaybackSpeed") or 1) - 1) > 1e-6
    except (TypeError, ValueError, AttributeError):
        return True


def _guid():
    return str(uuid.uuid4())


class Writer:
    def __init__(self, project_path):
        self.src = Path(project_path)
        try:
            self.root = ET.fromstring(P.read_xml(self.src))
        except (OSError, ET.ParseError):
            log_error("Writer.load")
            raise StoryError("StoryStudio couldn't open that Premiere project.")
        self.objs = {}
        for el in self.root:
            k = el.get("ObjectID") or el.get("ObjectUID")
            if k:
                self.objs[k] = el
        self.next_id = 1 + max([int(v) for el in self.root.iter() for a in ("ObjectID",
                                "ObjectRef") for v in [el.get(a)] if v and v.isdigit()] or [0])
        self._back = None

    # --- basics --------------------------------------------------------------
    def get(self, el, path):
        c = el.find(path) if el is not None else None
        if c is None:
            return None
        return self.objs.get(c.get("ObjectRef") or c.get("ObjectURef"))

    def new_key(self, numeric):
        if numeric:
            self.next_id += 1
            return str(self.next_id - 1)
        return _guid()

    def closure(self, start_key, allowed, skip_keys=()):
        out, stack = [], [start_key]
        while stack:
            k = stack.pop()
            if k in out or k in skip_keys or k not in self.objs:
                continue
            el = self.objs[k]
            if el.tag not in allowed:
                continue
            out.append(k)
            for c in el.iter():
                r = c.get("ObjectRef") or c.get("ObjectURef")
                if r:
                    stack.append(r)
        return out

    def clone(self, keys):
        """Deep-copy objects with fresh IDs; references among them are remapped."""
        remap = {}
        for k in keys:
            el = self.objs[k]
            remap[k] = self.new_key(el.get("ObjectID") is not None)
        new = {}
        for k in keys:
            el = copy.deepcopy(self.objs[k])
            if el.get("ObjectID") is not None:
                el.set("ObjectID", remap[k])
            else:
                el.set("ObjectUID", remap[k])
            for c in el.iter():
                for a in ("ObjectRef", "ObjectURef"):
                    v = c.get(a)
                    if v in remap:
                        c.set(a, remap[v])
            self.root.append(el)
            self.objs[remap[k]] = el
            new[k] = el
        return remap, new

    @staticmethod
    def setx(el, path, value):
        c = el.find(path)
        if c is None:
            parent_path, _, tag = path.rpartition("/")
            parent = el.find(parent_path) if parent_path else el
            if parent is None:
                return
            c = ET.SubElement(parent, tag)
        c.text = str(value)

    @staticmethod
    def drop(el, *tags):
        for t in tags:
            for c in list(el):
                if c.tag == t or re.fullmatch(t, c.tag):
                    el.remove(c)

    # --- finding things ------------------------------------------------------
    def sequence(self, uid=None):
        seqs = [el for el in self.root if el.tag == "Sequence"]
        if uid:
            for s in seqs:
                if (s.get("ObjectUID") or s.get("ObjectID")) == uid:
                    return s
        if not seqs:
            raise StoryError("That project has no sequence.")
        return seqs[0]

    def audio_group(self, seq):
        for g in seq.findall("TrackGroups/TrackGroup"):
            grp = self.get(g, "Second")
            if grp is not None and grp.tag == "AudioTrackGroup":
                return grp
        raise StoryError("That sequence has no audio tracks.")

    def template_item(self, want_ext=None):
        """An existing stereo, audio-only clip on any timeline to copy from."""
        best = []
        for el in self.root:
            if el.tag != "AudioClipTrackItem":
                continue
            sub = self.get(el, "ClipTrackItem/SubClip")
            clip = self.get(sub, "Clip") if sub is not None else None
            if clip is None:
                continue
            src = self.get(clip, "Clip/Source")
            if src is None or src.tag != "AudioMediaSource":
                continue
            media = self.get(src, "MediaSource/Media")
            if media is None or media.find("VideoStream") is not None:
                continue
            path = media.findtext("ActualMediaFilePath") or media.findtext("FilePath") or ""
            ext = Path(path).suffix.lower()
            if ext not in AUDIO_ONLY:
                continue
            if '"channellabel":101' not in (clip.findtext("AudioChannelLayout") or ""):
                continue                                   # stereo only
            chain = self.get(el, "ClipTrackItem/ComponentOwner/Components")
            plain = chain is not None and chain.findtext("DefaultVol") == "true"
            best.append((ext != want_ext, ext != ".wav", not plain, _speed_of(self, el), el,
                         ext))
        if not best:
            return None, None
        best.sort(key=lambda b: b[:4])
        return best[0][4], best[0][5]

    def spare_template(self):
        """A plain stereo .wav clip to copy from, for a project with no sound clip of its own
        (a picture-only timeline, such as a silent cartoon's rough cut). It is StoryStudio's
        own, kept in ss_audio_template.xml: its objects join this project under fresh IDs,
        and drop_spare() takes them out again before the project is saved. None when the
        project was made by an older Premiere than the template (it might not read it)."""
        from ss_core import APP_DIR
        try:
            t = ET.parse(APP_DIR / "ss_audio_template.xml").getroot()
            mine = max([int(e.get("Version")) for e in self.root
                        if e.tag == "Project" and (e.get("Version") or "").isdigit()] or [0])
            if mine and mine < int(t.get("ProjectVersion") or 0):
                return None
        except (OSError, ET.ParseError, ValueError):
            log_error("spare template")
            return None
        els = list(t)
        remap = {}
        for el in els:
            remap[el.get("ObjectID") or el.get("ObjectUID")] = \
                self.new_key(el.get("ObjectID") is not None)
        for el in els:
            old = el.get("ObjectID") or el.get("ObjectUID")
            el.set("ObjectID" if el.get("ObjectID") is not None else "ObjectUID", remap[old])
            for c in el.iter():
                for a in ("ObjectRef", "ObjectURef"):
                    if c.get(a) in remap:
                        c.set(a, remap[c.get(a)])
            self.root.append(el)
            self.objs[remap[old]] = el
        self._spare = [(remap[el_old], self.objs[remap[el_old]]) for el_old in remap]
        return self.objs.get(remap.get(t.get("Item")))

    def drop_spare(self):
        """Take StoryStudio's own template clip out again (the copies made from it stay)."""
        for key, el in getattr(self, "_spare", None) or []:
            if el in list(self.root):
                self.root.remove(el)
            self.objs.pop(key, None)
        self._spare = None

    # --- adding --------------------------------------------------------------
    def add_track(self, seq, name=None):
        grp = self.audio_group(seq)
        tracks = self.refs_list(grp, "TrackGroup/Tracks")
        if not tracks:
            raise StoryError("That sequence has no audio track to copy.")
        tmpl = self.objs[tracks[-1].get("ObjectURef") or tracks[-1].get("ObjectRef")]
        keys = self.closure(tmpl.get("ObjectUID") or tmpl.get("ObjectID"), TRACK_TAGS)
        remap, new = self.clone(keys)
        tr = new[keys[0]]
        tid = remap[keys[0]]
        # empty clip and transition lists
        for lst in tr.findall("ClipTrack/ClipItems/TrackItems") + \
                tr.findall("ClipTrack/TransitionItems/TrackItems"):
            for c in list(lst):
                lst.remove(c)
        idx = len(tracks)
        next_track_id = int(grp.findtext("TrackGroup/NextTrackID") or 100)
        self.setx(tr, "ClipTrack/Track/ID", next_track_id)
        self.setx(grp, "TrackGroup/NextTrackID", next_track_id + 1)
        for path in ("ClipTrack/Track/Index", "ClipTrack/ClipItems/Index",
                     "ClipTrack/TransitionItems/Index"):
            if tr.find(path) is not None:
                self.setx(tr, path, idx)
        if tr.find("AudioTrack/ID") is not None:
            self.setx(tr, "AudioTrack/ID", _guid())
        props = tr.find("ClipTrack/Track/Node/Properties")
        if props is not None:
            self.drop(props, "MZ.TrackTargeted")
            if name:
                self.setx(props, "TL.SQTrackName", name)
        lst = grp.find("TrackGroup/Tracks")
        ET.SubElement(lst, "Track", {"Index": str(idx), "ObjectURef": tid}) \
            if tr.get("ObjectUID") else ET.SubElement(lst, "Track", {"Index": str(idx),
                                                                     "ObjectRef": tid})
        # the mixer: master track's inlet lists every track as a source
        master = self.get(grp, "MasterTrack")
        inlet = self.get(master, "Inlet") if master is not None else None
        if inlet is not None and inlet.find("Sources") is not None:
            srcs = inlet.find("Sources")
            attr = "ObjectURef" if tr.get("ObjectUID") else "ObjectRef"
            ET.SubElement(srcs, "Source", {"Index": str(len(list(srcs))), attr: tid})
        # timeline patching lists (one entry per track)
        sp = seq.find("Node/Properties")
        if sp is not None:
            for c in sp:
                if c.tag.startswith("HSL.TimelinePatchingAndTargeting.AudioPatches") and \
                        (c.text or "").startswith("["):
                    c.text = c.text[:-1] + (',' if c.text.strip() != "[]" else "") + \
                        '{"mNumber":-1,"mState":0}]'
        return tr

    def refs(self, el, path):
        """Objects referenced by every element at path (in order)."""
        out = []
        if el is None:
            return out
        for c in el.findall(path):
            k = c.get("ObjectRef") or c.get("ObjectURef")
            if k in self.objs:
                out.append(self.objs[k])
        return out

    @staticmethod
    def refs_list(el, path):
        lst = el.find(path)
        return list(lst) if lst is not None else []

    def add_clip(self, track, template, file_path, start_sec, dur_sec, name, rate=48000,
                 bin_el=None, project_dir=None, src_in_sec=0.0, media_sec=None):
        """Put file_path on track at start_sec, dur_sec long, playing from src_in_sec into the
        file (media_sec: the file's whole length, when it is longer than what is used)."""
        keys = self.closure(template.get("ObjectID") or template.get("ObjectUID"), CLIP_TAGS)
        # the clip's project item (in a bin) points at its MasterClip
        sub = self.get(template, "ClipTrackItem/SubClip")
        mc_key = sub.find("MasterClip").get("ObjectURef") or sub.find("MasterClip").get(
            "ObjectRef")
        cpi_key = None
        for el in self.root:
            if el.tag == "ClipProjectItem":
                m = el.find("MasterClip")
                if m is not None and (m.get("ObjectURef") or m.get("ObjectRef")) == mc_key:
                    cpi_key = el.get("ObjectUID") or el.get("ObjectID")
                    break
        if cpi_key:
            keys.append(cpi_key)
        remap, new = self.clone(keys)
        D = int(round(dur_sec * TICKS))
        IN = int(round(src_in_sec * TICKS))
        MD = max(IN + D, int(round((media_sec or 0) * TICKS)))
        fr = str(int(round(TICKS / rate)))
        file_path = str(Path(file_path).resolve())
        fname = Path(file_path).name
        for k, el in new.items():
            t = el.tag
            if t == "AudioClipTrackItem":
                ti = el.find("ClipTrackItem/TrackItem")
                node = ti.find("Node")
                if node is not None:
                    ti.remove(node)
                s = int(round(start_sec * TICKS))
                for tag, v in (("Start", s), ("End", s + D)):
                    if ti.find(tag) is None:
                        c = ET.Element(tag)
                        ti.insert(0 if tag == "Start" else 1, c)
                    ti.find(tag).text = str(v)
                self.drop(el.find("ClipTrackItem"), "HeadTransition", "TailTransition")
                self.setx(el, "ID", _guid())
                item = el
            elif t == "SubClip" or t == "MasterClip":
                self.setx(el, "Name", name)
            elif t == "AudioClip":
                g = el.find("Gain")
                if g is not None:
                    g.text = "1.0"
                c = el.find("Clip")
                self.setx(c, "ClipID", _guid())
                _plain_speed(c)
                if c.find("OutPoint") is not None:
                    self.setx(c, "OutPoint", IN + D)
                    self.setx(c, "InPoint", IN)
                props = c.find("Node/Properties")
                if props is not None:
                    self.drop(props, "asl.clip.label.color", "asl.clip.label.name")
            elif t == "AudioMediaSource":
                self.setx(el, "OriginalDuration", MD)
            elif t == "Media":
                self.drop(el, "ModificationState", "ContentAndMetadataState",
                          r"MediaFileHistory\d+", "RelativePath")
                self.setx(el, "FilePath", file_path)
                self.setx(el, "ActualMediaFilePath", file_path)
                self.setx(el, "Title", fname)
                self.setx(el, "FileKey", _guid())
                if el.find("ConformedAudioRate") is not None:
                    self.setx(el, "ConformedAudioRate", fr)
                if project_dir:
                    try:
                        rel = os.path.relpath(file_path, project_dir)
                        c = ET.SubElement(el, "RelativePath")
                        c.text = rel
                    except ValueError:
                        pass
            elif t == "AudioStream":
                self.setx(el, "Duration", MD)
                self.setx(el, "FrameRate", fr)
                self.drop(el, "ConformedAudioPath", "PeakFilePath")
            elif t == "ClipLoggingInfo":
                self.setx(el, "ClipName", name)
                self.setx(el, "MediaInPoint", 0)
                self.setx(el, "MediaOutPoint", MD)
                self.setx(el, "MediaFrameRate", fr)
                self.drop(el, "ClipName_Logging")
            elif t == "ClipProjectItem":
                self.setx(el, "ProjectItem/Name", name)
                props = el.find("ProjectItem/Node/Properties")
                if props is not None:
                    self.drop(props, "project.icon.view.grid.order", "Column.PropertyText.Label")
                if bin_el is not None:
                    self.add_to_bin(bin_el, el)
        # put the new item on the track (an empty track has no TrackItems list yet)
        lst = track.find("ClipTrack/ClipItems/TrackItems")
        if lst is None:
            ci = track.find("ClipTrack/ClipItems")
            lst = ET.Element("TrackItems", {"Version": "1"})
            ci.insert(0, lst)
        ET.SubElement(lst, "TrackItem", {"Index": str(len(list(lst))),
                                         "ObjectRef": item.get("ObjectID")})
        return item

    # --- bins ----------------------------------------------------------------
    def root_bin(self):
        for el in self.root:
            if el.tag == "RootProjectItem":
                return el
        return None

    def add_to_bin(self, bin_el, item_el):
        items = bin_el.find("ProjectItemContainer/Items")
        if items is None:
            return
        attr = "ObjectURef" if item_el.get("ObjectUID") else "ObjectRef"
        ET.SubElement(items, "Item", {"Index": str(len(list(items))),
                                      attr: item_el.get("ObjectUID") or item_el.get("ObjectID")})

    def make_bin(self, name):
        root = self.root_bin()
        tmpl = next((el for el in self.root if el.tag == "BinProjectItem"), None)
        if root is None:
            return None
        if tmpl is None:
            return root                                   # no bin to copy: use the top level
        remap, new = self.clone([tmpl.get("ObjectUID") or tmpl.get("ObjectID")])
        b = list(new.values())[0]
        self.setx(b, "ProjectItem/Name", name)
        items = b.find("ProjectItemContainer/Items")
        if items is not None:
            for c in list(items):
                items.remove(c)
        props = b.find("ProjectItem/Node/Properties")
        if props is not None:
            self.drop(props, "project.icon.view.grid.order")
        if b.find("ProjectItem/Node/ID") is not None:
            ids = [int(x.text) for x in self.root.iter("ID")
                   if x.text and x.text.isdigit() and len(x.text) >= 7]
            self.setx(b, "ProjectItem/Node/ID", max(ids or [1000000]) + 1)
        if props is not None:
            for c in list(props):
                if c.tag == "ProjectViewState.ID":
                    c.text = _guid()
        self.add_to_bin(root, b)
        return b

    # --- swapping a clip's file for its cleaned copy ---------------------------
    def repoint(self, mapping):
        """mapping: {original file path: processed copy (same length and timing)}.
        Every Media object using an original now uses the copy; clips keep their in/out
        points, volume and effects. Premiere rebuilds its caches for the new file."""
        norm = {os.path.normcase(os.path.normpath(k)): v for k, v in mapping.items()}
        by_name = {}
        for k, v in mapping.items():
            by_name.setdefault(Path(k).name.lower(), []).append(v)
        done = 0
        for el in list(self.root):
            if el.tag != "Media":
                continue
            paths = [el.findtext(t) or "" for t in ("ActualMediaFilePath", "FilePath")]
            new = None
            for p in paths:
                if p and os.path.normcase(os.path.normpath(p)) in norm:
                    new = norm[os.path.normcase(os.path.normpath(p))]
                    break
            if new is None:                       # a file that moved and was found by name
                name = Path((paths[0] or paths[1]).replace("\\", "/")).name.lower()
                if name in by_name and len(by_name[name]) == 1 and \
                        not any(p and os.path.exists(p) for p in paths):
                    new = by_name[name][0]
            if new is None:
                continue
            new = str(Path(new).resolve())
            self.drop(el, "ModificationState", "ContentAndMetadataState",
                      r"MediaFileHistory\d+", "RelativePath")
            self.setx(el, "FilePath", new)
            self.setx(el, "ActualMediaFilePath", new)
            self.setx(el, "Title", Path(new).name)
            self.setx(el, "FileKey", _guid())
            stream = self.get(el, "AudioStream")
            if stream is not None:
                self.drop(stream, "ConformedAudioPath", "PeakFilePath")
            done += 1
        return done

    # --- giving one clip its own file ------------------------------------------
    def detach_items(self, files, project_dir=None):
        """files: {AudioClipTrackItem id: file}. Each clip gets its own copy of its media
        objects pointing at its own file (same length and timing as the original, so the
        clip's in/out points stay as they are). Used when one recording plays in several
        places and each place needs its own volume plan."""
        done = 0
        for item_id, path in files.items():
            el = self.objs.get(str(item_id))
            if el is None or el.tag != "AudioClipTrackItem":
                continue
            cti = el.find("ClipTrackItem")
            ref = cti.find("SubClip")
            sub_key = ref.get("ObjectRef") or ref.get("ObjectURef")
            tags = CLIP_TAGS - {"AudioClipTrackItem", "AudioComponentChain",
                                     "AudioFilterComponent", "AudioComponentParam"}
            keys = self.closure(sub_key, tags)
            sub = self.objs[sub_key]
            mc = sub.find("MasterClip")
            mc_key = mc.get("ObjectURef") or mc.get("ObjectRef")
            cpi = next((x for x in self.root if x.tag == "ClipProjectItem" and
                        (x.find("MasterClip") is not None) and
                        (x.find("MasterClip").get("ObjectURef") or
                         x.find("MasterClip").get("ObjectRef")) == mc_key), None)
            if cpi is not None:
                keys.append(cpi.get("ObjectUID") or cpi.get("ObjectID"))
            remap, new = self.clone(keys)
            name = Path(path).name
            fpath = str(Path(path).resolve())
            for k, x in new.items():
                if x.tag == "Media":
                    self.drop(x, "ModificationState", "ContentAndMetadataState",
                              r"MediaFileHistory\d+", "RelativePath")
                    self.setx(x, "FilePath", fpath)
                    self.setx(x, "ActualMediaFilePath", fpath)
                    self.setx(x, "Title", name)
                    self.setx(x, "FileKey", _guid())
                    if project_dir:
                        try:
                            c = ET.SubElement(x, "RelativePath")
                            c.text = os.path.relpath(fpath, project_dir)
                        except ValueError:
                            pass
                elif x.tag == "AudioStream":
                    self.drop(x, "ConformedAudioPath", "PeakFilePath")
                elif x.tag == "AudioClip":
                    self.setx(x.find("Clip"), "ClipID", _guid())
                elif x.tag in ("SubClip", "MasterClip"):
                    self.setx(x, "Name", Path(path).stem)
                elif x.tag == "ClipLoggingInfo":
                    self.setx(x, "ClipName", Path(path).stem)
                elif x.tag == "ClipProjectItem":
                    self.setx(x, "ProjectItem/Name", Path(path).stem)
                    root = self.root_bin()
                    if root is not None:
                        self.add_to_bin(root, x)
            attr = "ObjectRef" if self.objs[remap[sub_key]].get("ObjectID") else "ObjectURef"
            for a in ("ObjectRef", "ObjectURef"):
                ref.attrib.pop(a, None)
            ref.set(attr, remap[sub_key])
            done += 1
        return done

    # --- removing a clip's effects ---------------------------------------------
    ESP_KEEP = {"ESP.Tag"}

    def nudge_gain(self, changes):
        """changes: {AudioClipTrackItem id: dB to add}. Only the clip's Audio Gain changes
        (Premiere: right-click > Audio Gain); her effects, volume and keyframes stay."""
        done = 0
        for item_id, db in changes.items():
            el = self.objs.get(str(item_id))
            if el is None or el.tag != "AudioClipTrackItem" or abs(db) < 0.05:
                continue
            sub = self.get(el.find("ClipTrackItem"), "SubClip")
            clip = self.get(sub, "Clip") if sub is not None else None
            if clip is None:
                continue
            g = clip.find("Gain")
            try:
                old = float(g.text) if g is not None and g.text else 1.0
            except ValueError:
                old = 1.0
            if g is None:
                g = ET.SubElement(clip, "Gain")
            g.text = repr(float(old * 10 ** (db / 20)))
            done += 1
        return done

    def reset_items(self, levels):
        """levels: {AudioClipTrackItem id: clip volume in dB}. In StoryStudio's copy each of
        these clips loses its audio effects (Essential Sound noise reduction, limiter,
        compressor, Enhance Speech...) and its clip gain, and its volume is set to the
        given level, because StoryStudio's own cleanup now does that work."""
        done = 0
        for item_id, db in levels.items():
            el = self.objs.get(str(item_id))
            if el is None or el.tag != "AudioClipTrackItem":
                continue
            cti = el.find("ClipTrackItem")
            # Essential Sound settings on the clip
            props = cti.find("TrackItem/Node/Properties")
            if props is not None:
                for c in list(props):
                    if c.tag in self.ESP_KEEP:
                        continue
                    if c.tag.startswith(("ESP.", "MZ.EssentialSound", "MZ.AudioCategorization",
                                         "EnhanceSpeech")):
                        props.remove(c)
            # clip gain (Audio Gain / loudness auto-match) back to normal; Premiere's volume
            # control stops at +15 dB, so any boost beyond that goes into the clip gain
            sub = self.get(cti, "SubClip")
            clip = self.get(sub, "Clip") if sub is not None else None
            total = db
            extra = max(0.0, db - 15.0)
            db = min(db, 15.0)

            def set_gain(gain_db):
                if clip is None:
                    return False
                g = clip.find("Gain")
                if g is None and abs(gain_db) > 0.05:
                    g = ET.SubElement(clip, "Gain")
                if g is not None:
                    g.text = repr(float(10 ** (gain_db / 20)))
                return True
            set_gain(extra)
            # effects: keep only the built-in volume controls
            chain = self.get(cti, "ComponentOwner/Components")
            if chain is None:
                set_gain(total)
                done += 1
                continue
            comps = chain.find("ComponentChain/Components")
            level_set = False
            if comps is not None:
                keep = []
                for ref in list(comps):
                    comp = self.objs.get(ref.get("ObjectRef") or ref.get("ObjectURef"))
                    name = (comp.findtext("FilterMatchName") or "") if comp is not None else ""
                    if "Volume" in name:
                        keep.append(ref)
                        if name.startswith("Internal Volume"):
                            level_set = self._set_level(comp, db) or level_set
                    comps.remove(ref)
                for i, ref in enumerate(keep):
                    ref.set("Index", str(i))
                    comps.append(ref)
            if not level_set and abs(db) > 0.05:
                # no volume control on this clip (common on video clips, and on every clip in
                # newer Premiere projects): give it one, copied from another clip with the
                # same channel layout; failing that, the whole change goes into the clip gain
                if self._give_volume(cti, chain, db):
                    level_set = True
                elif set_gain(total):
                    level_set = True
                else:
                    log("couldn't set volume on clip", item_id, "(no volume control in it)")
            done += 1
        return done

    CHAIN_TAGS = {"AudioComponentChain", "AudioFilterComponent", "AudioComponentParam"}

    def _volume_template(self, layout):
        """An existing clip effect chain that has a volume control, same channel layout."""
        for el in self.root:
            if el.tag != "AudioComponentChain" or el.findtext("DefaultVol") == "true":
                continue
            if (el.findtext("AudioChannelLayout") or "") != layout:
                continue
            for comp in self.refs(el, "ComponentChain/Components/Component"):
                if (comp.findtext("FilterMatchName") or "").startswith("Internal Volume"):
                    return el
        return None

    def _give_volume(self, cti, old_chain, db):
        layout = old_chain.findtext("AudioChannelLayout") or ""
        tmpl = self._volume_template(layout)
        if tmpl is None:
            return False
        keys = self.closure(tmpl.get("ObjectID") or tmpl.get("ObjectUID"), self.CHAIN_TAGS)
        remap, new = self.clone(keys)
        chain = new[keys[0]]
        comps = chain.find("ComponentChain/Components")
        ok = False
        keep = []
        for ref in list(comps):
            comp = self.objs.get(ref.get("ObjectRef") or ref.get("ObjectURef"))
            name = (comp.findtext("FilterMatchName") or "") if comp is not None else ""
            if "Volume" in name:
                keep.append(ref)
                if name.startswith("Internal Volume"):
                    ok = self._set_level(comp, db) or ok
            comps.remove(ref)
        for i, ref in enumerate(keep):
            ref.set("Index", str(i))
            comps.append(ref)
        if not ok:
            return False
        holder = cti.find("ComponentOwner/Components")
        attr = "ObjectRef" if chain.get("ObjectID") else "ObjectURef"
        for a in ("ObjectRef", "ObjectURef"):
            holder.attrib.pop(a, None)
        holder.set(attr, remap[keys[0]])
        return True

    def _set_level(self, comp, db):
        for p in self.refs(comp, "AudioComponent/Component/Params/Param"):
            if p.findtext("Name") != "Level":
                continue
            v = P.db_to_level(db)
            self.setx(p, "CurrentValue", repr(float(v)))
            sk = p.find("StartKeyframe")
            if sk is not None and sk.text:
                parts = sk.text.split(",")
                if len(parts) > 1:
                    parts[1] = f"{v:.12f}"
                    sk.text = ",".join(parts)
            for tag in ("Keyframes", "IsTimeVarying"):
                c = p.find(tag)
                if c is not None:
                    if tag == "IsTimeVarying":
                        c.text = "false"
                    else:
                        p.remove(c)
            return True
        return False

    @staticmethod
    def refs_of(el):
        return el

    # --- saving --------------------------------------------------------------
    def save(self, out_path):
        data = ET.tostring(self.root, encoding="unicode")
        data = '<?xml version="1.0" encoding="UTF-8" ?>\n' + data
        tmp = Path(str(out_path) + ".tmp")
        with gzip.open(tmp, "wb", compresslevel=6) as f:
            f.write(data.encode("utf8"))
        os.replace(tmp, out_path)
        return str(out_path)


# ---------------------------------------------------------------------------
# retiming: trims, speed changes and removed shots, with the gaps closed
# ---------------------------------------------------------------------------
def _seq_tracks(w, seq):
    """Every track of the sequence: [{"kind", "items": [..], "trans": <TrackItems el>}], each
    item {"el", "cti", "id", "start", "end", "clip" (the inner Clip element), "in", "out",
    "speed", "media" (seconds of source, or None)}; plus the ticks in one frame."""
    tracks, frame = [], 0
    for g in seq.findall("TrackGroups/TrackGroup"):
        grp = w.get(g, "Second")
        if grp is None or grp.tag not in ("VideoTrackGroup", "AudioTrackGroup"):
            continue
        kind = "video" if grp.tag == "VideoTrackGroup" else "audio"
        if kind == "video":
            try:
                frame = int(grp.findtext("TrackGroup/FrameRate") or 0)
            except ValueError:
                frame = 0
        for tr in w.refs(grp, "TrackGroup/Tracks/Track"):
            ct = tr.find("ClipTrack")
            if ct is None:
                continue
            items = []
            for el in w.refs(ct, "ClipItems/TrackItems/TrackItem"):
                cti = el.find("ClipTrackItem")
                if cti is None or cti.find("TrackItem/End") is None:   # Start 0 is left out
                    continue
                sub = w.get(cti, "SubClip")
                clip = w.get(sub, "Clip") if sub is not None else None
                c = clip.find("Clip") if clip is not None else None
                if c is None:
                    continue

                def num(e, path, default=0):
                    try:
                        return int(float(e.findtext(path)))
                    except (TypeError, ValueError):
                        return default
                src = w.get(c, "Source")
                try:
                    speed = float(c.findtext("PlaybackSpeed") or 1.0)
                except ValueError:
                    speed = 1.0
                items.append({"el": el, "cti": cti, "c": c,
                              "id": el.get("ObjectID") or el.get("ObjectUID"),
                              "start": num(cti, "TrackItem/Start"),
                              "end": num(cti, "TrackItem/End"),
                              "in": num(c, "InPoint"), "out": num(c, "OutPoint"),
                              "has_out": c.find("OutPoint") is not None,
                              "speed": speed if speed > 0 else 1.0,
                              "media": num(src, "OriginalDuration", 0) if src is not None else 0,
                              "plain": c.find("PlaybackSpeed") is None or speed > 0})
            tracks.append({"kind": kind, "items": items, "ct": ct})
    return tracks, frame or int(TICKS / 24)


# ---------------------------------------------------------------------------
# links: sounds that move with their shot (Premiere's Clip > Link)
# ---------------------------------------------------------------------------
LINK_CLASS = "149d4ea5-a7d4-4b34-9bb7-16d783904bf2"
LINKED_TRACKS = ("StoryStudio effects",)       # ambience and music stay free


def _links_el(w, seq, make=False):
    """The sequence's <Links> list (Sequence/PersistentGroupContainer/LinkContainer/Links)."""
    pgc = seq.find("PersistentGroupContainer")
    if pgc is None:
        if not make:
            return None
        pgc = ET.Element("PersistentGroupContainer", {"Version": "1"})
        tg = seq.find("TrackGroups")
        seq.insert(list(seq).index(tg) if tg is not None else len(seq), pgc)
    lc = pgc.find("LinkContainer")
    if lc is None:
        if not make:
            return None
        lc = ET.SubElement(pgc, "LinkContainer", {"Version": "1"})
    links = lc.find("Links")
    if links is None and make:
        links = ET.SubElement(lc, "Links", {"Version": "1"})
    return links


def _link_members(w, seq):
    """{track item id: its Link object} for every link in the sequence."""
    out = {}
    links = _links_el(w, seq)
    for link in (w.refs(links, "Link") if links is not None else []):
        for r in link.findall("TrackItemGroup/TrackItems/TrackItem"):
            k = r.get("ObjectRef") or r.get("ObjectURef")
            if k:
                out[k] = link
    return out


def _unlink(w, seq, ids):
    """Take track items that left the timeline out of their links; a link left with one
    clip goes."""
    ids = set(ids)
    links = _links_el(w, seq)
    if not ids or links is None:
        return
    for ref in list(links):
        link = w.objs.get(ref.get("ObjectRef") or ref.get("ObjectURef"))
        lst = link.find("TrackItemGroup/TrackItems") if link is not None else None
        if lst is None:
            continue
        for r in list(lst):
            if (r.get("ObjectRef") or r.get("ObjectURef")) in ids:
                lst.remove(r)
        for i, r in enumerate(lst):
            r.set("Index", str(i))
        if len(lst) < 2:
            links.remove(ref)
    for i, r in enumerate(links):
        r.set("Index", str(i))


def _track_name(tr):
    return tr.findtext("ClipTrack/Track/Node/Properties/TL.SQTrackName") or ""


def link_sounds(w, seq, prefixes=LINKED_TRACKS):
    """Link every StoryStudio sound effect and character sound to the shot it belongs to (the
    picture clip it overlaps most in its first second and a half), the way Premiere's Clip >
    Link does. Moving, trimming or deleting a shot in Premiere then takes its sounds along.
    Background sounds and music stay free. Returns (sounds linked, shots)."""
    videos, sounds = [], []
    for g in seq.findall("TrackGroups/TrackGroup"):
        grp = w.get(g, "Second")
        if grp is None or grp.tag not in ("VideoTrackGroup", "AudioTrackGroup"):
            continue
        for ti, tr in enumerate(w.refs(grp, "TrackGroup/Tracks/Track")):
            ct = tr.find("ClipTrack")
            if ct is None:
                continue
            name = _track_name(tr)
            for el in w.refs(ct, "ClipItems/TrackItems/TrackItem"):
                cti = el.find("ClipTrackItem")
                key = el.get("ObjectID")
                if cti is None or not key:
                    continue
                try:
                    s = int(float(cti.findtext("TrackItem/Start") or 0))
                    e = int(float(cti.findtext("TrackItem/End")))
                except (TypeError, ValueError):
                    continue
                if grp.tag == "VideoTrackGroup" and el.tag == "VideoClipTrackItem":
                    videos.append((ti, s, e, key))
                elif grp.tag == "AudioTrackGroup" and el.tag == "AudioClipTrackItem" and \
                        name.startswith(tuple(prefixes)):
                    sounds.append((s, e, key))
    if not videos or not sounds:
        return 0, 0
    members = _link_members(w, seq)
    groups = {}                                    # video item id -> [audio item ids]
    for s, e, key in sounds:
        if key in members:                         # already linked (by her or before)
            continue
        hi = min(e, s + int(1.5 * TICKS))
        best, best_ov = None, 0
        for ti, vs, ve, vkey in videos:
            ov = min(hi, ve) - max(s, vs)            # lower video tracks win a tie
            if ov > 0 and (best is None or (ov, -ti) > (best_ov, -best[0])):
                best, best_ov = (ti, vkey), ov
        if best is None:                           # in a gap: the nearest shot before it
            before = [v for v in videos if v[2] <= s]
            if not before:
                continue
            best = (0, max(before, key=lambda v: v[2])[3])
        groups.setdefault(best[1], []).append(key)
    if not groups:
        return 0, 0
    links = _links_el(w, seq, make=True)
    n = 0
    for vkey, keys in groups.items():
        link = members.get(vkey)
        if link is None:
            lid = w.new_key(True)
            link = ET.Element("Link", {"ObjectID": lid, "ClassID": LINK_CLASS, "Version": "1"})
            tig = ET.SubElement(link, "TrackItemGroup", {"Version": "1"})
            lst = ET.SubElement(tig, "TrackItems", {"Version": "1"})
            ET.SubElement(lst, "TrackItem", {"Index": "0", "ObjectRef": vkey})
            w.root.append(link)
            w.objs[lid] = link
            ET.SubElement(links, "Link", {"Index": str(len(list(links))), "ObjectRef": lid})
        lst = link.find("TrackItemGroup/TrackItems")
        for k in keys:
            ET.SubElement(lst, "TrackItem", {"Index": str(len(list(lst))), "ObjectRef": k})
            n += 1
    return n, len(groups)


# ---------------------------------------------------------------------------
# lining the sounds up again after shots were moved, trimmed or deleted
# ---------------------------------------------------------------------------
def _num(e, path, default=0):
    try:
        return int(float(e.findtext(path)))
    except (TypeError, ValueError):
        return default


def _media_file(w, c):
    src = w.get(c, "Source")
    if src is None:
        return ""
    for r in src.iter():
        k = r.get("ObjectRef") or r.get("ObjectURef")
        m = w.objs.get(k) if k else None
        if m is not None and m.tag == "Media":
            return m.findtext("FilePath") or m.findtext("ActualMediaFilePath") or ""
    return ""


def _all_items(w, seq):
    """Every clip on the sequence, Start 0 included (Premiere leaves it out):
    [{"kind", "track" (element), "tname", "ti", "el", "cti", "c", "id", "start", "end", "in",
    "out", "speed", "file", "name"}]."""
    out = []
    for g in seq.findall("TrackGroups/TrackGroup"):
        grp = w.get(g, "Second")
        if grp is None or grp.tag not in ("VideoTrackGroup", "AudioTrackGroup"):
            continue
        kind = "video" if grp.tag == "VideoTrackGroup" else "audio"
        for ti, tr in enumerate(w.refs(grp, "TrackGroup/Tracks/Track")):
            ct = tr.find("ClipTrack")
            if ct is None:
                continue
            for el in w.refs(ct, "ClipItems/TrackItems/TrackItem"):
                cti = el.find("ClipTrackItem")
                if cti is None or cti.find("TrackItem/End") is None:
                    continue
                sub = w.get(cti, "SubClip")
                clip = w.get(sub, "Clip") if sub is not None else None
                c = clip.find("Clip") if clip is not None else None
                if c is None:
                    continue
                try:
                    speed = float(c.findtext("PlaybackSpeed") or 1.0) or 1.0
                except ValueError:
                    speed = 1.0
                out.append({"kind": kind, "track": tr, "tname": _track_name(tr), "ti": ti,
                            "el": el, "cti": cti, "c": c, "id": el.get("ObjectID"),
                            "start": _num(cti, "TrackItem/Start"),
                            "end": _num(cti, "TrackItem/End"),
                            "in": _num(c, "InPoint"), "out": _num(c, "OutPoint"),
                            "speed": abs(speed), "file": os.path.normcase(_media_file(w, c)),
                            "media": _num(w.get(c, "Source"), "OriginalDuration", 0)
                            if w.get(c, "Source") is not None else 0,
                            "name": sub.findtext("Name") or ""})
    return out


def _set_times(cti, start, end):
    ti = cti.find("TrackItem")
    for tag, v in (("Start", start), ("End", end)):
        e = ti.find(tag)
        if e is None:
            e = ET.Element(tag)
            node = ti.find("Node")
            at = (list(ti).index(node) + 1) if node is not None else 0
            ti.insert(at if tag == "Start" else list(ti).index(ti.find("Start")) + 1, e)
        e.text = str(int(v))


def _anchor(sound, videos):
    """The shot a sound belongs to: the picture clip it overlaps most in its first 1.5 s."""
    s, e = sound["start"], sound["end"]
    hi = min(e, s + int(1.5 * TICKS))
    best = None
    for v in videos:
        ov = min(hi, v["end"]) - max(s, v["start"])
        if ov > 0 and (best is None or (ov, -v["ti"]) > (best[0], -best[1]["ti"])):
            best = (ov, v)
    if best:
        return best[1]
    before = [v for v in videos if v["end"] <= s]
    return max(before, key=lambda v: v["end"]) if before else None


SOUND_GROUPS = (("StoryStudio effects", False), ("StoryStudio ambience", True),
                ("StoryStudio music", True))         # track name -> long layer (a "bed")?


def _group_of(tname):
    for k, (prefix, bed) in enumerate(SOUND_GROUPS):
        if tname.startswith(prefix):
            return k
    return None


def _shot_at(t, videos):
    """The picture clip playing at time t (lowest video track first)."""
    best = None
    for v in videos:
        if v["start"] <= t < v["end"] and (best is None or v["ti"] < best["ti"]):
            best = v
    return best


def line_up_sounds(project_path, seq_uid, ref_path, out_path, prefixes=None,
                   ref_sounds=None, groups=None, skip=()):
    """After shots were moved, trimmed or deleted in Premiere: put every StoryStudio sound back
    with its shot. Effects and character sounds keep their exact moment in their shot (a sound
    whose moment was trimmed away is shortened or goes; a deleted shot's sounds go with it).
    Background sounds and music (long layers) start where the shot they started on is now and
    end where the shot they ended on is now, so they follow their scene.
    ref_path / ref_sounds: the timeline as it was when the sounds were placed, and
    [{"name", "start", "end", "bed"}] in seconds; groups: [(ref_path, ref_sounds)] when the
    sounds were made in several runs, each on the timeline as it was then.
    Returns {"project", "moved", "removed", "shots_moved", "kept", "beds_moved"}."""
    if groups is None:
        groups = [(ref_path, ref_sounds)]
    w = Writer(project_path)
    seq = w.sequence(seq_uid)
    now = _all_items(w, seq)
    T = TICKS
    nvid = [i for i in now if i["kind"] == "video"]
    nsnd = [i for i in now if i["kind"] == "audio" and _group_of(i["tname"]) is not None]
    by_file = {}
    for v in nvid:
        by_file.setdefault(v["file"], []).append(v)
    pool, moved_shots = {}, 0
    for g_path, g_sounds in groups:
        rw = Writer(g_path)
        ref = _all_items(rw, rw.sequence(seq_uid))
        rvid = [i for i in ref if i["kind"] == "video"]
        if g_sounds is not None:
            rsnd = [{"name": x["name"], "start": int(round(x["start"] * TICKS)),
                     "end": int(round(x["end"] * TICKS)), "bed": bool(x.get("bed")),
                     "file": x.get("file", "")} for x in g_sounds]
        else:
            rsnd = [dict(i, bed=SOUND_GROUPS[_group_of(i["tname"])][1]) for i in ref
                    if i["kind"] == "audio" and _group_of(i["tname"]) is not None]
        # each shot then -> the same shot now (same clip file; nearest in point if used twice)
        used, shot_now = set(), {}
        for v in sorted(rvid, key=lambda v: v["start"]):
            cands = [x for x in by_file.get(v["file"], []) if x["id"] not in used]
            if not cands:
                continue
            x = min(cands, key=lambda x: (abs(x["in"] - v["in"]), abs(x["start"] - v["start"])))
            used.add(x["id"])
            shot_now[v["id"]] = x
        moved_shots = max(moved_shots, sum(1 for v in rvid if v["id"] in shot_now and
                                           abs(shot_now[v["id"]]["start"] - v["start"]) > T // 50))
        for r in rsnd:
            pool.setdefault(r["name"], []).append((r, rvid, shot_now))
    if not pool:
        raise StoryError("StoryStudio couldn't find its sounds for this timeline.")

    def moment(t, rvid, shot_now):
        """Where the moment that was at time t (on the old timeline) is now, or None."""
        v = _shot_at(t, rvid)
        x = shot_now.get(v["id"]) if v else None
        if x is None:
            return None
        src = v["in"] + (t - v["start"]) * v["speed"]
        return x["start"] + (src - x["in"]) / x["speed"], x

    plan, gone, beds_moved = [], [], 0
    claimed, restored, extra = set(), 0, 0
    cur = {}
    for s in nsnd:
        cur.setdefault(s["name"], []).append(s)
    tmpl = None
    group_track = {}
    for s in nsnd:
        group_track.setdefault(_group_of(s["tname"]), s["track"])

    def take(name, near):
        """Her clip of this sound nearest to a time (each one used once)."""
        cs = [c for c in cur.get(name, []) if c["id"] not in claimed]
        if not cs:
            return None
        c = min(cs, key=lambda c: abs(c["start"] - near))
        claimed.add(c["id"])
        return c

    order = sorted(((r, rvid, sn) for lst in pool.values() for r, rvid, sn in lst),
                   key=lambda e: e[0]["start"])
    for r, rvid, shot_now in order:
        if r.get("bed"):
            s = take(r["name"], r["start"])
            if s is None:
                continue
            # a long layer: its start follows the shot it started on, its end the shot it
            # ended on; it keeps playing from the same place in its own sound
            a = moment(r["start"] + T // 25, rvid, shot_now)
            if a is None:
                continue                           # its first shot is gone: leave it be
            b = moment(r["end"] - T // 25, rvid, shot_now)
            new_start = a[0] - T // 25
            length = r["end"] - r["start"]
            new_end = b[0] + T // 25 if b and b[0] > new_start + T else new_start + length
            room = (s["media"] - s["in"]) / s["speed"] if s.get("media") else length
            new_end = min(new_end, new_start + max(room, T))
            new_start = max(0, new_start)
            if abs(new_start - s["start"]) > T // 10 or abs(new_end - s["end"]) > T // 10:
                beds_moved += 1
            plan.append((s, int(round(new_start)), int(round(new_end)), s["in"]))
            continue
        v = _anchor(r, rvid)
        if v is None:
            s = take(r["name"], r["start"])        # can't tell its shot: leave it be
            continue
        # the sound's stretch of its shot's clip, and every piece of that clip on the timeline
        sa = v["in"] + (r["start"] - v["start"]) * v["speed"]
        length = r["end"] - r["start"]
        sb = sa + length * v["speed"]
        v_end = v["in"] + (v["end"] - v["start"]) * v["speed"]
        tail_over = max(0, r["end"] - v["end"])          # it ran past its cut before
        pieces = []
        for x in by_file.get(v["file"], []):
            xa, xb = x["in"], x["in"] + (x["end"] - x["start"]) * x["speed"]
            oa, ob = max(sa, xa), min(sb, xb)
            if (ob - oa) / x["speed"] < T * 0.15:
                continue
            start = x["start"] + (oa - xa) / x["speed"]
            end = x["start"] + (ob - xa) / x["speed"]
            if sb > v_end and abs(xb - v_end) < T // 20:  # it may run past this cut again
                end += tail_over
            pieces.append((start, end, (oa - sa) / v["speed"]))
        if not pieces:                             # its shot (or that part) is gone
            s = take(r["name"], r["start"])
            if s is not None:
                gone.append(s)
            continue
        placed = 0
        for start, end, head in sorted(pieces):
            s = take(r["name"], start)
            if s is None:                          # missing, or a second piece of the shot
                if r["name"] in skip or not r.get("file") or not os.path.exists(r["file"]):
                    continue
                tr = group_track.get(0)
                if tr is None:
                    continue
                if tmpl is None:
                    tmpl, _ = w.template_item(".wav")
                    if tmpl is None:
                        continue
                try:
                    from ss_core import probe
                    media = probe(r["file"])["duration"] or 0
                except Exception:
                    media = 0
                item = w.add_clip(tr, tmpl, r["file"], start / T, (end - start) / T, r["name"],
                                  project_dir=str(Path(out_path).parent), src_in_sec=head / T,
                                  media_sec=media)
                s = {"id": item.get("ObjectID"), "track": tr, "tname": SOUND_GROUPS[0][0],
                     "cti": item.find("ClipTrackItem"), "c": None, "in": int(head),
                     "start": int(start), "end": int(end), "speed": 1.0, "name": r["name"]}
                claimed.add(s["id"])
                nsnd.append(s)
                if placed:
                    extra += 1                     # a second piece of the same shot
                else:
                    restored += 1                  # it had gone missing
                placed += 1
                continue
            plan.append((s, int(round(start)), int(round(end)), int(round(head))))
            placed += 1
    # move them
    moved = 0
    for s, a, b, src_in in plan:
        if abs(a - s["start"]) > T // 100 or abs(b - s["end"]) > T // 100:
            moved += 1
        _set_times(s["cti"], a, b)
        if s["c"] is not None and s["c"].find("InPoint") is not None:
            Writer.setx(s["c"], "InPoint", int(src_in))
        if s["c"] is not None and s["c"].find("OutPoint") is not None:
            Writer.setx(s["c"], "OutPoint", int(src_in) + int((b - a) * s["speed"]))
        s["start"], s["end"] = a, b
    # lay each kind of track out again so no two clips share a moment on one track
    gone_ids = {s["id"] for s in gone}
    for k, (prefix, _) in enumerate(SOUND_GROUPS):
        mine = [s for s in nsnd if _group_of(s["tname"]) == k]
        if not mine:
            continue
        tracks = []
        for s in mine:
            if s["track"] not in tracks:
                tracks.append(s["track"])
        ids = {s["id"] for s in mine}
        for tr in tracks:
            lst = tr.find("ClipTrack/ClipItems/TrackItems")
            for ref_el in list(lst):
                if (ref_el.get("ObjectRef") or ref_el.get("ObjectURef")) in ids:
                    lst.remove(ref_el)
        lanes = []
        for tr in tracks:                          # anything else still on a track blocks it
            rest = [w.objs.get(e.get("ObjectRef") or e.get("ObjectURef"))
                    for e in tr.find("ClipTrack/ClipItems/TrackItems")]
            lanes.append({"track": tr, "items": [],
                          "end": max([_num(r, "ClipTrackItem/TrackItem/End") for r in rest
                                      if r is not None] or [-1])})
        for s in sorted([s for s in mine if s["id"] not in gone_ids], key=lambda s: s["start"]):
            lane = next((l for l in lanes if l["end"] <= s["start"]), None)
            if lane is None:
                lane = {"track": w.add_track(seq, f"{prefix} {len(lanes) + 1}"), "end": -1,
                        "items": []}
                lanes.append(lane)
            lane["items"].append(s)
            lane["end"] = s["end"]
        for lane in lanes:
            lst = lane["track"].find("ClipTrack/ClipItems/TrackItems")
            if lst is None:
                ci = lane["track"].find("ClipTrack/ClipItems")
                lst = ET.Element("TrackItems", {"Version": "1"})
                ci.insert(0, lst)
            for s in lane["items"]:
                ET.SubElement(lst, "TrackItem", {"Index": "0", "ObjectRef": s["id"]})
            items = sorted(list(lst), key=lambda e: _num(w.objs[e.get("ObjectRef")],
                                                         "ClipTrackItem/TrackItem/Start"))
            for e in items:
                lst.remove(e)
            for n, e in enumerate(items):
                e.set("Index", str(n))
                lst.append(e)
    _unlink(w, seq, list(gone_ids))
    w.save(out_path)
    kept = len(nsnd) - len(gone)
    log("lined up sounds", out_path, "moved", moved, "removed", len(gone), "shots moved",
        moved_shots, "beds moved", beds_moved, "put back", restored, "second pieces", extra)
    return {"project": str(out_path), "moved": moved, "removed": len(gone),
            "shots_moved": moved_shots, "kept": kept, "beds_moved": beds_moved,
            "restored": restored, "extra": extra}


def write_linked(project_path, seq_uid, out_path):
    """A copy of the project with every StoryStudio sound effect linked to its shot."""
    w = Writer(project_path)
    seq = w.sequence(seq_uid)
    n, shots = link_sounds(w, seq)
    if not n:
        raise StoryError("StoryStudio didn't find any of its sound effects on this timeline "
                         "that aren't linked already. Open the copy ending in “(with sound)”.")
    w.save(out_path)
    log("wrote linked project", out_path, n, "sounds on", shots, "shots")
    return {"project": str(out_path), "sounds": n, "shots": shots}


def retime(project_path, seq_uid, edits, out_path):
    """Save a copy of the project with timing changes made on the sequence's video shots and
    every gap they leave closed up (everything later moves earlier, sound included).

    edits: {video track item id: {"head": seconds to trim off the start, "tail": seconds off
    the end, "speed": percent of the current speed (100 = leave), "more": seconds to hold
    longer, "cut": True to remove the shot}}.
    Returns {"project", "changed", "skipped": [reasons], "shorter": seconds}."""
    w = Writer(project_path)
    seq = w.sequence(seq_uid)
    tracks, frame = _seq_tracks(w, seq)
    T = float(TICKS)

    def snap(t):                                   # timeline times sit on whole frames
        return int(round(t / frame)) * frame
    old_end = max([i["end"] for tr in tracks for i in tr["items"]] or [0])
    done, skipped, plans = 0, [], {}
    for tr in tracks:
        if tr["kind"] != "video":
            continue
        for it in tr["items"]:
            e = edits.get(str(it["id"]))
            if not e:
                continue
            length = it["end"] - it["start"]
            if not it["has_out"] or it["out"] <= it["in"] or length <= 0:
                skipped.append("a shot StoryStudio can't safely retime (a title or still)")
                continue
            sp = (it["out"] - it["in"]) / length          # how the shot really plays now
            if e.get("cut"):
                plans[it["id"]] = {"cut": True, "len": 0}
                done += 1
                continue
            head = max(0.0, float(e.get("head") or 0)) * T
            tail = max(0.0, float(e.get("tail") or 0)) * T
            if head + tail > length - 0.3 * T:           # never trim a shot away to nothing
                k = max(0.0, (length - 0.3 * T)) / (head + tail)
                head, tail = head * k, tail * k
            new_in, new_out = it["in"] + head * sp, it["out"] - tail * sp
            new_sp = sp * max(0.25, min(4.0, float(e.get("speed") or 100) / 100.0))
            new_len = (new_out - new_in) / new_sp
            more = max(0.0, float(e.get("more") or 0)) * T
            if more:
                spare = max(0.0, (it["media"] - new_out) / new_sp) if it["media"] else 0.0
                use = min(more, spare)
                new_out += use * new_sp
                new_len += use
                rest = more - use
                if rest > 0.04 * T:                      # no more footage: play it a bit slower
                    slow = max(0.6, new_len / (new_len + rest))
                    new_sp *= slow
                    new_len = (new_out - new_in) / new_sp
            plans[it["id"]] = {"in": new_in, "out": new_out, "speed": new_sp, "len": new_len,
                               "head": head, "old_speed": sp}
            done += 1
    if not done:
        raise StoryError("None of the changes could be made on this project.")
    # where every moment of the old timeline lands on the new one
    edited = sorted(((it["start"], it["end"], plans[it["id"]]) for tr in tracks
                     if tr["kind"] == "video" for it in tr["items"] if it["id"] in plans),
                    key=lambda x: x[0])

    def where(t):
        shift = 0.0
        for a, b, p in edited:
            if t >= b:
                shift += p["len"] - (b - a)
            elif t > a:                                  # inside a changed shot
                if p.get("cut"):
                    return a + shift
                inside = (t - a - p["head"]) * p["old_speed"] / p["speed"]
                return a + shift + max(0.0, min(p["len"], inside))
            else:
                break
        return t + shift
    members = _link_members(w, seq)               # sounds linked to a cut shot go with it
    linked_cut = set()
    for k, link in members.items():
        if plans.get(k, {}).get("cut"):
            linked_cut |= {r.get("ObjectRef") for r in
                           link.findall("TrackItemGroup/TrackItems/TrackItem")}
    for tr in tracks:
        keep = []
        for it in tr["items"]:
            if tr["kind"] == "audio" and it["id"] in linked_cut:
                it["gone"] = True
                continue
            p = plans.get(it["id"]) if tr["kind"] == "video" else None
            if p is None and tr["kind"] == "audio":       # a shot's own sound follows its shot
                p = next((q for a, b, q in edited if abs(a - it["start"]) < frame and
                          abs(b - it["end"]) < frame and not q.get("cut") and
                          it["has_out"]), None)
                if p is not None:
                    sp = (it["out"] - it["in"]) / max(1, it["end"] - it["start"])
                    k = p["speed"] / p["old_speed"]
                    p = {"in": it["in"] + p["head"] * sp, "speed": sp * k, "len": p["len"],
                         "out": it["in"] + p["head"] * sp + p["len"] * sp * k}
            if p is not None and p.get("cut"):
                it["gone"] = True
                continue
            it["new_start"] = snap(where(it["start"]))
            if p is not None:
                it["new_end"] = max(it["new_start"] + frame, snap(where(it["start"]) + p["len"]))
                it["new_in"], it["new_speed"] = int(round(p["in"])), p["speed"]
            else:
                it["new_end"] = it["new_start"] + (it["end"] - it["start"])
            keep.append(it)
        keep.sort(key=lambda i: i["new_start"])
        for a, b in zip(keep, keep[1:]):                 # two clips never share a moment
            if a["new_end"] > b["new_start"]:
                a["new_end"] = max(a["new_start"] + frame, b["new_start"])
                if a["new_end"] > b["new_start"]:        # no room left at all: it goes
                    a["gone"] = True
        for it in keep:
            if it.get("gone"):
                continue
            cti, c = it["cti"], it["c"]
            _set_times(cti, it["new_start"], it["new_end"])
            new_len = it["new_end"] - it["new_start"]
            if "new_in" in it:
                sp = it["new_speed"]
                while it["media"] and it["new_in"] + new_len * sp > it["media"] and \
                        new_len > frame:
                    new_len -= frame
                    w.setx(cti, "TrackItem/End", it["new_start"] + new_len)
                    it["new_end"] = it["new_start"] + new_len
                w.setx(c, "InPoint", it["new_in"])
                w.setx(c, "OutPoint", int(round(it["new_in"] + new_len * sp)))
                if abs(sp - 1.0) < 0.005:
                    w.drop(c, "PlaybackSpeed")
                elif c.find("PlaybackSpeed") is not None:
                    w.setx(c, "PlaybackSpeed", round(sp, 6))
                else:                                    # goes just before the in point
                    el = ET.Element("PlaybackSpeed")
                    el.text = str(round(sp, 6))
                    c.insert(list(c).index(c.find("InPoint")), el)
            elif it["has_out"] and new_len != it["end"] - it["start"]:
                sp = (it["out"] - it["in"]) / max(1, it["end"] - it["start"])
                w.setx(c, "OutPoint", int(round(it["in"] + new_len * sp)))
        # removed shots leave the track, with their transitions
        gone = [it for it in tr["items"] if it.get("gone")]
        lst = tr["ct"].find("ClipItems/TrackItems")
        tlst = tr["ct"].find("TransitionItems/TrackItems")
        dead = set()
        for it in gone:
            for ref in list(lst):
                if (ref.get("ObjectRef") or ref.get("ObjectURef")) == it["id"]:
                    lst.remove(ref)
            for tag in ("HeadTransition", "TailTransition"):
                r = it["cti"].find(tag)
                if r is not None:
                    dead.add(r.get("ObjectRef") or r.get("ObjectURef"))
        # transitions stay with their cut: a clip's opening one starts where the clip now
        # starts, its closing one ends where the clip now ends
        cuts = {}
        for it in tr["items"]:
            if it.get("gone"):
                continue
            for tag, t in (("HeadTransition", it["new_start"]), ("TailTransition", it["new_end"])):
                r = it["cti"].find(tag)
                if r is not None:
                    key = r.get("ObjectRef") or r.get("ObjectURef")
                    cuts[key] = t
                    dead.discard(key)                    # shared with a clip that stays
        if tlst is not None:
            for ref in list(tlst):
                key = ref.get("ObjectRef") or ref.get("ObjectURef")
                tel = w.objs.get(key)
                if key in dead or tel is None:
                    if key in dead:
                        tlst.remove(ref)
                    continue
                ti = tel.find("TransitionTrackItem")
                if ti is None or key not in cuts:
                    continue
                try:
                    s0 = int(float(ti.findtext("TrackItem/Start") or 0))
                    e0 = int(float(ti.findtext("TrackItem/End")))
                    al = int(float(ti.findtext("Alignment") or 0))
                except (TypeError, ValueError):
                    continue
                start = max(0, cuts[key] - al)
                _set_times(ti, start, start + (e0 - s0))
            for i, ref in enumerate(list(tlst)):
                if ref.get("Index") is not None:
                    ref.set("Index", str(i))
        if lst is not None:
            for i, ref in enumerate(list(lst)):
                if ref.get("Index") is not None:
                    ref.set("Index", str(i))
    new_end = max([i["new_end"] for tr in tracks for i in tr["items"]
                   if not i.get("gone") and "new_end" in i and tr["kind"] == "video"] or [0])
    old_video_end = max([i["end"] for tr in tracks if tr["kind"] == "video"
                         for i in tr["items"]] or [0])
    _unlink(w, seq, [i["id"] for tr in tracks for i in tr["items"] if i.get("gone")])
    w.save(out_path)
    log("wrote retimed project", out_path, "changed", done, "skipped", len(skipped))
    return {"project": str(out_path), "changed": done, "skipped": skipped,
            "shorter": round((old_video_end - new_end) / T, 2)}


MARKER_CLASS = "a45508e0-3ff7-4d04-90a7-2e0dfff4c910"
# Premiere's marker colours (stored as 0xAABBGGRR); green is Premiere's default (no colour)
MARKER_COLOURS = {"red": 4281740498, "orange": 4280578025, "yellow": 4281049552,
                  "blue": 4294741314, "green": None}


def add_markers(w, seq, markers):
    """Sequence markers (the coloured flags on the timeline ruler). markers: dicts with
    start (s), optional end (s), name, comment, colour (see MARKER_COLOURS)."""
    import json
    holder = w.get(seq, "MarkerOwner/Markers")
    if holder is None:
        holder = w.get(seq, "Markers")
    if holder is None:
        # a sequence that never had a marker: make its marker list the way Premiere does
        oid = w.new_key(True)
        holder = ET.SubElement(w.root, "Markers", {
            "ObjectID": oid, "ClassID": "bee50706-b524-416c-9f03-b596ce5f6866",
            "Version": "4"})
        ET.SubElement(holder, "ByGUID").text = "byGUID"
        ET.SubElement(holder, "LastMetadataState").text = "00000000-0000-0000-0000-000000000000"
        ET.SubElement(holder, "LastContentState").text = "00000000-0000-0000-0000-000000000000"
        w.objs[oid] = holder
        owner = ET.Element("MarkerOwner", {"Version": "1"})
        ET.SubElement(owner, "Markers", {"ObjectRef": oid})
        node = seq.find("Node")
        seq.insert(list(seq).index(node) + 1 if node is not None else 0, owner)
    lst = holder.find("Markers")
    if lst is None:
        lst = ET.Element("Markers", {"Version": "1"})
        holder.insert(0, lst)
    n = len(lst.findall("Marker"))
    for m in markers:
        guid = _guid()
        d = {"mComment": m.get("comment", ""), "mMarkerID": guid, "mName": m.get("name", ""),
             "mStartTime": {"ticks": int(round(max(0.0, m["start"]) * TICKS))},
             "mType": "Comment"}
        if m.get("end") and m["end"] > m["start"] + 0.05:
            d["mDuration"] = {"ticks": int(round((m["end"] - m["start"]) * TICKS))}
        col = MARKER_COLOURS.get(m.get("colour") or "green")
        if col is not None:
            d["mCuePointList"] = [{"mKey": "keywordExtDVAv1_ce447bd7-ff30-49a8-ab20-518fafba0c7e",
                                   "mValue": json.dumps({"color": col})}]
        oid = w.new_key(True)
        obj = ET.SubElement(w.root, "Marker", {"ObjectID": oid, "ClassID": MARKER_CLASS,
                                               "Version": "3"})
        ET.SubElement(obj, "DVAMarker").text = json.dumps({"DVAMarker": d},
                                                          separators=(",", ":"))
        w.objs[oid] = obj
        it = ET.SubElement(lst, "Marker", {"Version": "1", "Index": str(n)})
        ET.SubElement(it, "First").text = guid
        ET.SubElement(it, "Second", {"ObjectRef": oid})
        n += 1
    for tag in ("LastMetadataState", "LastContentState"):
        if holder.find(tag) is not None:
            holder.find(tag).text = "00000000-0000-0000-0000-000000000000"
    return n


def write_markers(project_path, seq_uid, markers, out_path):
    """Save a copy of the project with the markers added to the sequence."""
    w = Writer(project_path)
    seq = w.sequence(seq_uid)
    add_markers(w, seq, markers)
    w.save(out_path)
    log("wrote markers", len(markers), out_path)
    return str(out_path)


def write_project(project_path, seq_uid, tracks, out_path, sound_dir, repoint=None,
                  reset=None, detach=None, link=False):
    """tracks: [(track_name, [ {file, start, duration, name}, ...]), ...].
    repoint: {original audio file: cleaned copy} for the storyteller's own narration/music.
    Saves a copy of the project with these tracks added to the sequence. Returns out_path."""
    w = Writer(project_path)
    seq = w.sequence(seq_uid)
    if repoint:
        n = w.repoint(repoint)
        log("repointed", n, "media to cleaned copies")
    if detach:
        n = w.detach_items(detach, project_dir=str(Path(out_path).parent))
        log("gave", n, "clips their own balanced file")
    if reset:
        n = w.reset_items(reset)
        log("removed effects / set volume on", n, "clips")
    tmpl, ext = w.template_item(".wav")
    if tmpl is None:                         # a picture-only timeline: StoryStudio's own clip
        tmpl, ext = w.spare_template(), ".wav"
    if tmpl is None:
        raise StoryError("This project has no sound clip on any timeline for StoryStudio to "
                         "copy, so it couldn't add the sound tracks. In Premiere, put any "
                         "sound file on the timeline (even a short one), save, and run this "
                         "again. Your new sounds are in the StoryStudio folder meanwhile.")
    b = w.make_bin("StoryStudio sounds")
    for tname, clips in tracks:
        if not clips:
            continue
        tr = w.add_track(seq, tname)
        for c in clips:
            w.add_clip(tr, tmpl, c["file"], c["start"], c["duration"], c["name"], bin_el=b,
                       project_dir=str(Path(out_path).parent))
    if link:
        try:
            n, shots = link_sounds(w, seq)
            log("linked", n, "sounds to", shots, "shots")
        except Exception:
            log_error("link sounds")
    w.drop_spare()
    w.save(out_path)
    log("wrote project", out_path, "template ext", ext)
    return str(out_path), ext


# ---------------------------------------------------------------------------
# video: a new top video track with one clip (karaoke lyrics overlay)
# ---------------------------------------------------------------------------
VIDEO_CLIP_TAGS = {"VideoClipTrackItem", "SubClip", "MasterClip", "VideoClip",
                   "VideoMediaSource", "Media", "VideoStream", "ClipChannelGroupVectorSerializer",
                   "ClipChannelVectorSerializer", "ClipChannelSerializer", "ClipLoggingInfo",
                   "Markers", "VideoComponentChain", "VideoFilterComponent",
                   "VideoComponentParam", "PointComponentParam", "ArbVideoComponentParam",
                   "SecondaryContent"}
VIDEO_TRACK_TAGS = {"VideoClipTrack", "VideoComponentChain", "VideoFilterComponent",
                    "VideoComponentParam", "PointComponentParam"}
VIDEO_FILES = {".mp4", ".mov", ".m4v", ".mxf", ".avi", ".mkv", ".webm"}


def _video_group(w, seq):
    for g in seq.findall("TrackGroups/TrackGroup"):
        grp = w.get(g, "Second")
        if grp is not None and grp.tag == "VideoTrackGroup":
            return grp
    raise StoryError("That sequence has no video tracks.")


def _video_template(w):
    """A video-only clip (a shot with no sound) to copy from, one at normal speed if any."""
    found = None
    for el in w.root:
        if el.tag != "VideoClipTrackItem":
            continue
        sub = w.get(el, "ClipTrackItem/SubClip")
        clip = w.get(sub, "Clip") if sub is not None else None
        src = w.get(clip, "Clip/Source") if clip is not None else None
        if src is None or src.tag != "VideoMediaSource":
            continue
        media = w.get(src, "MediaSource/Media")
        if media is None or media.find("AudioStream") is not None or \
                media.find("VideoStream") is None:
            continue
        path = media.findtext("ActualMediaFilePath") or media.findtext("FilePath") or ""
        if Path(path).suffix.lower() in VIDEO_FILES:
            if not _speed_of(w, el):
                return el                      # a shot at normal speed: best to copy
            found = found if found is not None else el
    return found


def _set_param(w, comp, name, value):
    for p in w.refs(comp, ".//Params/Param"):
        if p.findtext("Name") != name:
            continue
        sk = p.find("StartKeyframe")
        if sk is not None and sk.text:
            parts = sk.text.split(",")
            if len(parts) > 1:
                parts[1] = value
                sk.text = ",".join(parts)
        if p.find("CurrentValue") is not None:
            p.find("CurrentValue").text = value
        c = p.find("Keyframes")
        if c is not None:
            p.remove(c)
        if p.find("IsTimeVarying") is not None:
            p.find("IsTimeVarying").text = "false"
        return True
    return False


def add_overlay(project_path, seq_uid, file_path, start, dur, width, height, fps, out_path,
                name="Karaoke lyrics"):
    """Save a copy of the project with file_path (a video with a see-through background) on
    a new video track above everything, at start, for dur seconds, at 100% size."""
    w = Writer(project_path)
    seq = w.sequence(seq_uid)
    tmpl = _video_template(w)
    if tmpl is None:
        raise StoryError("This project has no video clip StoryStudio can copy, so it can not "
                         "add the lyrics track directly.")
    grp = _video_group(w, seq)
    tracks = w.refs_list(grp, "TrackGroup/Tracks")
    ttmpl = w.objs[tracks[-1].get("ObjectURef") or tracks[-1].get("ObjectRef")]
    keys = w.closure(ttmpl.get("ObjectUID") or ttmpl.get("ObjectID"), VIDEO_TRACK_TAGS)
    remap, new = w.clone(keys)
    tr = new[keys[0]]
    for lst in tr.findall("ClipTrack/ClipItems/TrackItems") + \
            tr.findall("ClipTrack/TransitionItems/TrackItems"):
        for c in list(lst):
            lst.remove(c)
    idx = len(tracks)
    nid = int(grp.findtext("TrackGroup/NextTrackID") or 100)
    w.setx(tr, "ClipTrack/Track/ID", nid)
    w.setx(grp, "TrackGroup/NextTrackID", nid + 1)
    for path in ("ClipTrack/Track/Index", "ClipTrack/ClipItems/Index",
                 "ClipTrack/TransitionItems/Index"):
        if tr.find(path) is not None:
            w.setx(tr, path, idx)
    props = tr.find("ClipTrack/Track/Node/Properties")
    if props is not None:
        w.drop(props, "MZ.TrackTargeted")
    attr = "ObjectURef" if tr.get("ObjectUID") else "ObjectRef"
    ET.SubElement(grp.find("TrackGroup/Tracks"), "Track", {"Index": str(idx),
                                                          attr: remap[keys[0]]})
    sp = seq.find("Node/Properties")
    if sp is not None:
        for c in sp:
            if c.tag.startswith("HSL.TimelinePatchingAndTargeting.VideoPatches") and \
                    (c.text or "").startswith("["):
                c.text = c.text[:-1] + ("," if c.text.strip() != "[]" else "") + \
                    '{"mNumber":-1,"mState":0}]'
    # the clip
    keys = w.closure(tmpl.get("ObjectID") or tmpl.get("ObjectUID"), VIDEO_CLIP_TAGS)
    sub = w.get(tmpl, "ClipTrackItem/SubClip")
    mc_key = sub.find("MasterClip").get("ObjectURef") or sub.find("MasterClip").get("ObjectRef")
    cpi = next((x for x in w.root if x.tag == "ClipProjectItem" and
                x.find("MasterClip") is not None and
                (x.find("MasterClip").get("ObjectURef") or
                 x.find("MasterClip").get("ObjectRef")) == mc_key), None)
    if cpi is not None:
        keys.append(cpi.get("ObjectUID") or cpi.get("ObjectID"))
    remap, new = w.clone(keys)
    D = int(round(dur * TICKS))
    fr = str(int(round(TICKS / fps)))
    fpath = str(Path(file_path).resolve())
    item = None
    for k, el in new.items():
        t = el.tag
        if t == "VideoClipTrackItem":
            ti = el.find("ClipTrackItem/TrackItem")
            node = ti.find("Node")
            if node is not None:
                ti.remove(node)
            s0 = int(round(start * TICKS))
            for tag, v in (("Start", s0), ("End", s0 + D)):
                if ti.find(tag) is None:
                    ti.insert(0 if tag == "Start" else 1, ET.Element(tag))
                ti.find(tag).text = str(v)
            w.drop(el.find("ClipTrackItem"), "HeadTransition", "TailTransition")
            if el.find("FrameRect") is not None:
                w.setx(el, "FrameRect", f"0,0,{width},{height}")
            item = el
        elif t in ("SubClip", "MasterClip"):
            w.setx(el, "Name", name)
        elif t == "VideoClip":
            c = el.find("Clip")
            w.setx(c, "ClipID", _guid())
            _plain_speed(c)                    # the lyrics must play at 100%, or they drift
            if c.find("OutPoint") is not None:
                w.setx(c, "OutPoint", D)
                w.setx(c, "InPoint", 0)
        elif t == "VideoMediaSource":
            w.setx(el, "OriginalDuration", D)
        elif t == "Media":
            w.drop(el, "ModificationState", "ContentAndMetadataState", r"MediaFileHistory\d+",
                   "RelativePath")
            w.setx(el, "FilePath", fpath)
            w.setx(el, "ActualMediaFilePath", fpath)
            w.setx(el, "Title", Path(fpath).name)
            w.setx(el, "FileKey", _guid())
            try:
                ET.SubElement(el, "RelativePath").text = os.path.relpath(
                    fpath, str(Path(out_path).parent))
            except ValueError:
                pass
        elif t == "VideoStream":
            w.setx(el, "Duration", D)
            w.setx(el, "FrameRate", fr)
            w.setx(el, "FrameRect", f"0,0,{width},{height}")
            w.setx(el, "IgnoreAlpha", "false")              # the lyrics sit over the movie
            w.drop(el, "CodecType", "OriginalColorSpace", "AlphaType")
        elif t == "ClipLoggingInfo":
            w.setx(el, "ClipName", name)
            w.setx(el, "MediaInPoint", 0)
            w.setx(el, "MediaOutPoint", D)
            w.setx(el, "MediaFrameRate", fr)
        elif t == "ClipProjectItem":
            w.setx(el, "ProjectItem/Name", name)
            root = w.root_bin()
            if root is not None:
                w.add_to_bin(root, el)
        elif t == "VideoFilterComponent" and "Motion" in "".join(
                (x.text or "") for x in el.iter("MatchName")):
            # 100% size, centred, no crop (the shot this was copied from may be scaled)
            for pname, val in (("Scale", "100."), ("Scale Width", "100."),
                               ("Position", "0.5:0.5"), ("Anchor Point", "0.5:0.5"),
                               ("Rotation", "0."), ("Crop Left", "0."), ("Crop Top", "0."),
                               ("Crop Right", "0."), ("Crop Bottom", "0.")):
                _set_param(w, el, pname, val)
    lst = tr.find("ClipTrack/ClipItems/TrackItems")
    if lst is None:
        lst = ET.Element("TrackItems", {"Version": "1"})
        tr.find("ClipTrack/ClipItems").insert(0, lst)
    ET.SubElement(lst, "TrackItem", {"Index": str(len(list(lst))),
                                     "ObjectRef": item.get("ObjectID")})
    w.save(out_path)
    log("wrote karaoke project", out_path)
    return str(out_path)
