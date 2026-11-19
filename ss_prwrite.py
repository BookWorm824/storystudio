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
                 bin_el=None, project_dir=None):
        """Put file_path on track at start_sec (whole file, dur_sec long)."""
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
                    self.setx(c, "OutPoint", D)
                    self.setx(c, "InPoint", 0)
                props = c.find("Node/Properties")
                if props is not None:
                    self.drop(props, "asl.clip.label.color", "asl.clip.label.name")
            elif t == "AudioMediaSource":
                self.setx(el, "OriginalDuration", D)
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
                self.setx(el, "Duration", D)
                self.setx(el, "FrameRate", fr)
                self.drop(el, "ConformedAudioPath", "PeakFilePath")
            elif t == "ClipLoggingInfo":
                self.setx(el, "ClipName", name)
                self.setx(el, "MediaInPoint", 0)
                self.setx(el, "MediaOutPoint", D)
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
                if cti is None or cti.find("TrackItem/Start") is None:
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
    for tr in tracks:
        keep = []
        for it in tr["items"]:
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
            w.setx(cti, "TrackItem/Start", it["new_start"])
            w.setx(cti, "TrackItem/End", it["new_end"])
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
                    s0 = int(float(ti.findtext("TrackItem/Start")))
                    e0 = int(float(ti.findtext("TrackItem/End")))
                    al = int(float(ti.findtext("Alignment") or 0))
                except (TypeError, ValueError):
                    continue
                start = max(0, cuts[key] - al)
                w.setx(ti, "TrackItem/Start", start)
                w.setx(ti, "TrackItem/End", start + (e0 - s0))
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
                  reset=None, detach=None):
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
    if tmpl is None:
        raise StoryError("This project has no music or voice clip StoryStudio can copy, so it "
                         "can't add tracks to it directly. Use the import file instead.")
    b = w.make_bin("StoryStudio sounds")
    for tname, clips in tracks:
        if not clips:
            continue
        tr = w.add_track(seq, tname)
        for c in clips:
            w.add_clip(tr, tmpl, c["file"], c["start"], c["duration"], c["name"], bin_el=b,
                       project_dir=str(Path(out_path).parent))
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
