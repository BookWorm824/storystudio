"""Story check: does the story come across?

    audit(st, settings, progress, cancel)      the whole movie
    ask(st, text, at, progress, cancel)        "Tell StoryStudio what you'd like better"

1. A first-time viewer: the AI watches stills of the whole movie WITHOUT the script and says,
   moment by moment, what it understands is happening, what each character seems to feel and
   want, and where it gets unsure or lost. Not knowing the script is the point: it can't fill
   gaps with what it was told, the way the storyteller can't help doing.
2. The comparison: that retelling is set against the script, the shot list (what each shot is
   for, its prompts) and the sounds on the timeline. Every place where the story doesn't come
   across becomes a note with fixes of five kinds: change the edit, add a sound, make a clip
   again (a new image-to-video prompt), add a new shot (text-to-image and image-to-video
   prompts), or change the words.
3. Fixes are handed back ready to use: prompts to copy, a sound that goes straight into the
   project's sound list, and new or rewritten shots saved into the shot list. Every note is also
   a marker on the timeline in a "(story check)" copy of the project.

The storyteller can also describe what she sees ("at 1:20 it looks like he's dancing, not
pulling"); the AI looks closely at that moment and answers with the same kinds of fixes.
"""
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ss_core import StoryError, fmt_time, log, log_error, read_story, safe_name
import ss_media as M
import ss_premiere as P

FIX_TYPES = ["edit", "sound", "redo_clip", "new_shot", "words"]
FIX_NAMES = {"edit": "Change the edit", "sound": "Add a sound", "redo_clip": "Make the clip again",
             "new_shot": "Add a new shot", "words": "Change the words"}
SOUND_KIND = {"effect": "action", "character": "voice", "music": "music"}
MAX_ASKS = 40


def _camera():
    import ss_script as SB
    return SB.CAMERA


def kind_text(kind):
    return {"music": "music video",
            "cartoon": "silent slapstick cartoon (no words: the pictures, the characters' faces "
                       "and the sound effects tell it)"}.get(kind, "story / short movie")


# ---------------------------------------------------------------------------
# what is on the timeline, and the shot list it was made from
# ---------------------------------------------------------------------------
def timeline_shots(seq):
    """The pictures on the timeline in order: [{"start", "end", "file", "name"}], with pieces
    of the same clip joined back together."""
    out = []
    for a, b, it in P.picture_segments(seq):
        if it is None or b - a < 0.05:
            continue
        f = it.get("file") or ""
        if out and out[-1]["file"] == f and abs(out[-1]["end"] - a) < 0.02 and \
                out[-1]["item"] == it.get("id"):
            out[-1]["end"] = b
            continue
        out.append({"start": a, "end": b, "file": f, "item": it.get("id"),
                    "name": it.get("name") or Path(f).stem})
    for s in out:
        s.pop("item", None)
    return out


def _norm(p):
    return os.path.normcase(os.path.normpath(str(p))) if p else ""


def _shot_paths(s):
    ps = set(_norm(p) for p in (s.get("paths") or []))
    ps.add(_norm(s.get("best")))
    ps |= {_norm(f.get("path")) for f in (s.get("files") or [])}
    ps.discard("")
    return ps


def find_shot_list(tl):
    """The shot list this edit was made from, matched by the clip files on the timeline (or by
    shot numbers in the file names, in the shot list's own folder). Returns (path, list,
    {timeline index: shot n}) or (None, None, {})."""
    import ss_script as SB
    import ss_place as PL
    files = [_norm(s["file"]) for s in tl]
    best = (None, None, {})
    try:
        recent = SB.recent(40)
    except Exception:
        log_error("story check: shot lists")
        return best
    for rec in recent:
        try:
            r = json.loads(Path(rec["path"]).read_text(encoding="utf8"))
        except (OSError, ValueError):
            continue
        shots = r.get("shots") or []
        by_path = {}
        for s in shots:
            for p in _shot_paths(s):
                by_path.setdefault(p, s["n"])
        folder = _norm(r.get("clip_folder"))
        by_label = {str(s.get("label") or s["n"]).lower(): s["n"] for s in shots}
        mapping = {}
        for i, f in enumerate(files):
            if f in by_path:
                mapping[i] = by_path[f]
            elif folder and f.startswith(folder):
                lab = PL.label_of(Path(f).name)
                if lab and lab[0].lower() in by_label:
                    mapping[i] = by_label[lab[0].lower()]
        if len(mapping) >= max(2, 0.3 * len(files)) and len(mapping) > len(best[2]):
            best = (rec["path"], r, mapping)
    return best


def timeline_sounds(seq):
    """The sounds on the timeline with plain names: [(start, end, name)]. StoryStudio's own
    sounds are named after what they are ("c012 - Chester gasps.wav")."""
    _, aud = P.flatten(seq)
    seen, out = set(), []
    for a in aud:
        f = a.get("file") or ""
        if not f:
            continue
        name = re.sub(r"^c\d{3}\s*-\s*", "", Path(f).stem)
        name = re.sub(r"\s*\((take|v)\s*\d+\)$", "", name, flags=re.I).strip() or Path(f).stem
        key = (round(float(a["start"]), 1), name.lower())
        if key in seen:                         # the same sound on its left and right tracks
            continue
        seen.add(key)
        out.append((float(a["start"]), float(a["end"]), name))
    out.sort()
    return out


def _cast(r):
    if not r or not r.get("characters"):
        return "(no recurring characters listed)"
    return "\n".join(f"- {c['name']}: {c.get('short') or c.get('look', '')}"
                     for c in r["characters"])


def _shot_of(r, n):
    return next((s for s in (r or {}).get("shots", []) if s["n"] == n), None)


def shot_label(s):
    return str(s.get("label") or s["n"])


def _shot_listing(tl, r, mapping, lo=None, hi=None):
    rows = []
    for i, t in enumerate(tl):
        if lo is not None and (t["end"] < lo or t["start"] > hi):
            continue
        s = _shot_of(r, mapping.get(i))
        head = f"{fmt_time(t['start'])}-{fmt_time(t['end'])} ({t['start']:.1f}-{t['end']:.1f}s)"
        if not s:
            rows.append(f"{head} clip \"{t['name']}\" (not in the shot list)")
            continue
        bits = [f"{head} SHOT {shot_label(s)}" + (f" \"{s['title']}\"" if s.get("title") else "")]
        if s.get("purpose"):
            bits.append(f"  meant to show: {s['purpose']}")
        if s.get("picture") and s.get("picture") != s.get("purpose"):
            bits.append(f"  picture: {s['picture'][:300]}")
        if s.get("motion"):
            bits.append(f"  action prompt: {s['motion'][:300]}")
        if s.get("words"):
            bits.append(f"  {'sounds planned' if (r or {}).get('kind') == 'cartoon' else 'words'}: "
                        f"{s['words'][:200]}")
        rows.append("\n".join(bits))
    return "\n".join(rows) or "(none)"


def _sound_listing(sounds, lo=None, hi=None, limit=400):
    rows = [f"{fmt_time(a)} {name}" for a, b, name in sounds
            if lo is None or (b >= lo and a <= hi)]
    return "\n".join(rows[:limit]) or "(no sounds on the timeline)"


# ---------------------------------------------------------------------------
# the schemas
# ---------------------------------------------------------------------------
def fix_schema():
    return {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": FIX_TYPES},
            "title": {"type": "string", "description": "2 to 6 plain words"},
            "what": {"type": "string", "description": "the step, plainly, in one or two "
                     "sentences: for an edit, exactly what to trim, move, hold or swap and by "
                     "how much; for a sound, what the viewer should hear and what it tells them"},
            "shot": {"type": "string", "description": "the shot number it is about, as listed "
                     "(\"6\", \"20.1\"); for new_shot, the shot the new one comes AFTER; empty "
                     "if no shot"},
            "at": {"type": "number", "description": "seconds from the start of the movie where "
                   "it happens (for a sound: where the sound starts)"},
            "end": {"type": "number", "description": "seconds where it ends (a sound: where "
                    "it ends; most are 0.5 to 3 seconds)"},
            "sound": {"type": "string", "description": "for type sound: a short prompt for a "
                      "sound-effect AI, 8-25 words: what makes the sound, how, and the feeling "
                      "(a character's sound is a wordless noise: gasp, squeak, giggle, grunt, "
                      "sigh, never words or singing). Empty for other types"},
            "sound_kind": {"type": "string", "enum": ["effect", "character", "music", ""],
                           "description": "for type sound: effect (including short musical "
                           "stings, dings and fanfares), a character's own wordless sound, or "
                           "music (only a background tune lasting many seconds); empty for "
                           "other types"},
            "picture": {"type": "string", "description": "for new_shot (and redo_clip when "
                        "the picture itself must change): the text-to-image prompt; empty "
                        "otherwise"},
            "motion": {"type": "string", "description": "for redo_clip and new_shot: the "
                       "image-to-video prompt; empty otherwise"},
            "camera": {"type": "string", "enum": [""] + _camera(),
                       "description": "for redo_clip and new_shot: the MiniMax camera command "
                       "matching the motion; empty otherwise"},
        },
        "required": ["type", "title", "what", "shot", "at", "end", "sound", "sound_kind",
                     "picture", "motion", "camera"],
        "additionalProperties": False,
    }


VIEWER = {
    "type": "object",
    "properties": {
        "moments": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "time": {"type": "number", "description": "seconds from the start"},
                "end": {"type": "number", "description": "seconds"},
                "happens": {"type": "string", "description": "what a viewer understands "
                            "happens here, in one plain sentence, including what a character "
                            "seems to feel or want"},
                "clear": {"type": "string", "enum": ["clear", "unsure", "lost"],
                          "description": "clear = a viewer gets it; unsure = they can guess "
                          "but aren't sure; lost = they can't tell what is going on or why"},
                "wondering": {"type": "string", "description": "what a viewer would be "
                              "wondering or getting wrong here; empty when clear"},
            },
            "required": ["time", "end", "happens", "clear", "wondering"],
            "additionalProperties": False}},
        "story": {"type": "string", "description": "the story of this part as the viewer "
                  "would tell a friend, 2-4 sentences"},
    },
    "required": ["moments", "story"],
    "additionalProperties": False,
}


def note_schema():
    return {
        "type": "object",
        "properties": {
            "time": {"type": "number"}, "end": {"type": "number"},
            "severity": {"type": "string", "enum": ["fix", "suggest"],
                         "description": "fix = viewers will miss or misread something that "
                         "matters; suggest = it would read better"},
            "kind": {"type": "string", "enum": ["unclear", "missing", "misread", "too fast",
                                                 "payoff", "feeling", "who", "where"]},
            "title": {"type": "string", "description": "2 to 6 plain words"},
            "problem": {"type": "string", "description": "what the story means to say here "
                        "and what a viewer gets instead, one or two plain sentences"},
            "fixes": {"type": "array", "items": fix_schema(),
                      "description": "1 to 3 ways to fix it, the easiest first"},
        },
        "required": ["time", "end", "severity", "kind", "title", "problem", "fixes"],
        "additionalProperties": False,
    }


def compare_schema():
    return {
        "type": "object",
        "properties": {
            "comes_across": {"type": "string", "enum": ["yes", "mostly", "partly", "no"]},
            "overall": {"type": "string", "description": "3-4 warm, honest, plain sentences: "
                        "does the story come across, and where it gets lost"},
            "working": {"type": "array", "items": {"type": "string"},
                        "description": "2-4 moments that tell the story well, with times"},
            "top_fixes": {"type": "array", "items": {"type": "string"},
                          "description": "the 1-3 changes that would help the story most"},
            "notes": {"type": "array", "items": note_schema()},
        },
        "required": ["comes_across", "overall", "working", "top_fixes", "notes"],
        "additionalProperties": False,
    }


def ask_schema():
    return {
        "type": "object",
        "properties": {
            "reply": {"type": "string", "description": "2-4 plain, kind sentences: what you "
                      "see at that moment, why it looks that way, and what will fix it"},
            "fixes": {"type": "array", "items": fix_schema(),
                      "description": "1 to 4 fixes, the one most likely to work first"},
            "question": {"type": "string", "description": "one short question if you truly "
                         "need her answer to help; otherwise empty"},
        },
        "required": ["reply", "fixes", "question"],
        "additionalProperties": False,
    }


VIEWER_SYSTEM = """You are an ordinary viewer watching a short video for the very first time. \
For a family cartoon, imagine a bright 7-year-old watching with a parent. You have NOT read the \
script and nobody has told you the story: you only know what the pictures (and any words) show.

You see stills taken at regular times, with the time printed in yellow. Retell, moment by \
moment, what you understand is happening, who is who, where they are, what each character \
seems to feel and want, and why they do what they do. Be honest: when you can't tell what a \
character is doing or why, when something seems to come from nowhere, when a joke's set-up is \
missing so the joke doesn't land, or when a moment goes by too fast to take in, say so \
("unsure" or "lost") and say what you'd be wondering. Don't invent a story to fill the gaps; \
a gap you notice is the most useful thing you can report. Plain words, no film jargon."""

ADVISOR_SYSTEM = """You are a kind, experienced story editor and director helping a storyteller \
who makes children's stories, silent cartoons and music videos with AI pictures (ChatGPT or Nano \
Banana), AI video clips (MiniMax Hailuo, image to video) and Adobe Premiere. She is not a \
professional: write in plain, warm, everyday words, no jargon, and give her exact things to do.

Your job is STORY CLARITY: making sure a first-time viewer understands what happens and why, \
feels what the characters feel, and gets every joke and the ending. Judge only that (glitches, \
colour and audio levels are checked by other tools).

Ways to fix a moment that doesn't come across (offer the easiest that will really work):
- edit: trim, hold a shot longer so it can be read, swap two shots, move a shot, cut a shot that \
confuses. Holding a reaction a second longer is often all it takes.
- sound: a sound that tells the viewer what they can't see or what a character feels (a gasp of \
delight, a tummy rumble, a creak before something breaks, a "ta-da" sting). Characters NEVER \
speak: their sounds are wordless noises.
- redo_clip: make the same shot again with a clearer action. Write the new image-to-video prompt.
- new_shot: add a shot the story is missing (a close-up on a face to show a feeling, a look at \
the thing a character sees, an establishing shot of the place, a cutaway that sets up a joke). \
Write its text-to-image prompt and its image-to-video prompt.
- words: change or add narration, a line of lyrics or a title card (never for a silent cartoon).

Prompt rules:
TEXT-TO-IMAGE: {picture}
IMAGE-TO-VIDEO: {motion}

Studio rules: no religious symbols or churches; Jesus or God only ever as a warm light; no \
Christmas, Halloween, Easter or St. Patrick's Day imagery."""


def _advisor_system():
    import ss_script as SB
    return ADVISOR_SYSTEM.format(picture=SB.PICTURE_RULES, motion=SB.MOTION_RULES)


def _clean_fix(f, labels):
    t = f.get("type") if f.get("type") in FIX_TYPES else "edit"
    shot = str(f.get("shot") or "").strip()
    shot = re.sub(r"^(shot|extra shot)\s+", "", shot, flags=re.I)
    try:
        at = max(0.0, float(f.get("at") or 0))
        end = max(at, float(f.get("end") or at))
    except (TypeError, ValueError):
        at = end = 0.0
    out = {"type": t, "title": (f.get("title") or FIX_NAMES[t]).strip(),
           "what": (f.get("what") or "").strip(), "shot": shot if shot in labels else "",
           "at": round(at, 2), "end": round(end, 2),
           "sound": (f.get("sound") or "").strip() if t == "sound" else "",
           "sound_kind": f.get("sound_kind") or ("effect" if t == "sound" else ""),
           "picture": (f.get("picture") or "").strip() if t in ("redo_clip", "new_shot") else "",
           "motion": (f.get("motion") or "").strip() if t in ("redo_clip", "new_shot") else "",
           "camera": f.get("camera") or ""}
    if t in ("redo_clip", "new_shot") and out["motion"] and not out["camera"]:
        out["camera"] = "Static shot"
    if t == "sound" and not out["sound"]:
        out["sound"] = out["what"]
    if t == "sound" and out["end"] < out["at"] + 0.3:
        out["end"] = round(out["at"] + (1.2 if out["sound_kind"] == "character" else 1.5), 2)
    return out


# ---------------------------------------------------------------------------
# getting ready: the picture, the sound, the words, the shot list
# ---------------------------------------------------------------------------
def prepare(st, progress=None, cancel=None, words=True):
    """Everything both the story check and her own notes need. Kept in st["audit"]."""
    import storysound as E
    import ss_feedback as F
    say = progress or (lambda f, m: None)
    work = Path(st["work"]) / "story"
    work.mkdir(parents=True, exist_ok=True)
    seq = E._sequence(st)
    st.setdefault("width", seq["width"])
    st.setdefault("height", seq["height"])
    dur = seq["duration"]
    if not st.get("picture") or not Path(st["picture"]).exists():
        if st.get("export") and Path(st["export"]).exists():
            st["picture"] = st["export"]
        else:
            say(0.01, "Getting the picture ready…")
            st["picture"] = M.build_picture(seq, Path(st["work"]), lambda f: say(
                0.01 + 0.15 * f, "Getting the picture ready…"))
    if cancel and cancel():
        raise StoryError("Stopped.")
    tl = timeline_shots(seq)
    path, r, mapping = find_shot_list(tl)
    sounds = timeline_sounds(seq)
    kind = (r or {}).get("kind") if (r or {}).get("kind") in ("story", "music", "cartoon") else ""
    lines, preview = [], st["picture"]
    say(0.17, "Listening to the sound…")
    try:
        clips = M.build_stems(seq, work, lambda f: say(0.17 + 0.08 * f, "Listening to the sound…"))
        mixwav = work / "timeline_mix.wav"
        M.mix([dict(c, gain_db=c.get("gain_db", 0.0) + c.get("track_db", 0.0)) for c in clips],
              dur, mixwav)
        if not kind:
            kind = F.guess_kind(seq, clips)
        mono = M.wav_mono16k(mixwav)
        if words and kind != "cartoon":
            say(0.26, "Listening to the words…")
            lines = M.lines_from_words(M.transcribe_raw(mono, lambda f: say(
                0.26 + 0.08 * f, "Listening to the words…")))
            if kind == "story" and sum(len(l["text"].split()) for l in lines) < 6:
                kind = "cartoon"
        prev = work / "story check preview.mp4"
        M.mux(st["picture"], str(mixwav), str(prev))
        preview = str(prev)
    except StoryError:
        raise
    except Exception:
        log_error("story check sound")
    story = ""
    if st.get("story"):
        try:
            story = read_story(st["story"])
        except StoryError:
            story = ""
    if not story and r:
        story = r.get("script") or ""
    a = st.setdefault("audit", {})
    a.update({"kind": kind or "story", "shots": tl, "shot_list": path or "",
              "mapping": {str(k): v for k, v in mapping.items()}, "sounds": sounds,
              "lines": lines, "preview": preview, "duration": dur,
              "story_text": story[:30000], "title": st["seq_name"]})
    a.setdefault("notes", [])
    a.setdefault("asks", [])
    return a


def _shot_list(a):
    if not a.get("shot_list"):
        return None
    try:
        return json.loads(Path(a["shot_list"]).read_text(encoding="utf8"))
    except (OSError, ValueError):
        return None


def _mapping(a):
    return {int(k): v for k, v in (a.get("mapping") or {}).items()}


def _labels(a, r):
    labels = {shot_label(s) for s in (r or {}).get("shots", [])}
    return labels


# ---------------------------------------------------------------------------
# 1. the first-time viewer
# ---------------------------------------------------------------------------
def viewer_pass(st, a, sheets, every, progress=None, cancel=None, per_request=6, workers=3):
    import threading
    import ss_claude as C
    groups = [sheets[i:i + per_request] for i in range(0, len(sheets), per_request)]
    dur = a["duration"]
    parts, lock, done = {}, threading.Lock(), [0]

    def one(gi):
        grp = groups[gi]
        if cancel and cancel():
            return
        lo = grp[0][0]
        hi = groups[gi + 1][0][0] if gi + 1 < len(groups) else dur
        heard = "\n".join(f"{fmt_time(l['s'])}: {l['text']}" for l in a.get("lines", [])
                          if l["e"] >= lo and l["s"] <= hi)
        prompt = f"""You are watching a {kind_text(a['kind'])}, {dur:.0f} seconds long, called \
"{a['title']}". This is {f'part {gi + 1} of {len(groups)}: ' if len(groups) > 1 else ''}\
{lo:.0f}s to {hi:.0f}s.
{"WHAT IS SAID OR SUNG (speech recognition, may mishear words):" + chr(10) + heard if heard else
 "Nothing is said in this part." if a['kind'] != 'cartoon' else
 "It has no words; you'd also hear sound effects, but here you only have the pictures."}

The stills are one every {every:g} second(s), with the time in yellow. Retell what you \
understand, moment by moment (one moment for every beat of the action, usually every 2 to 6 \
seconds), with times in seconds from the start of the video."""
        images = [(f"Stills {fmt_time(x)} to {fmt_time(y)}:", p) for x, y, p in grp]
        try:
            res = C.ask(prompt, VIEWER, system=VIEWER_SYSTEM, images=images, timeout=900)
        except StoryError as e:
            log("story check: viewer part skipped:", e)
            res = {}
        with lock:
            parts[gi] = res
            done[0] += 1
            if progress:
                progress(done[0] / len(groups), f"{C.ai_name()} is watching it as a first-time "
                         f"viewer ({done[0]} of {len(groups)})…")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, range(len(groups))))
    if not parts or not any(parts.values()):
        raise StoryError(f"{C.ai_name()} couldn't watch the video this time. Try again in a "
                         "few minutes.")
    moments, story = [], []
    for gi in sorted(parts):
        res = parts[gi] or {}
        for m in res.get("moments", []):
            try:
                t = float(m["time"])
                e = max(t, float(m.get("end") or t))
            except (TypeError, ValueError, KeyError):
                continue
            moments.append({"time": round(t, 2), "end": round(e, 2),
                            "happens": (m.get("happens") or "").strip(),
                            "clear": m.get("clear") if m.get("clear") in ("clear", "unsure",
                                                                           "lost") else "unsure",
                            "wondering": (m.get("wondering") or "").strip()})
        if res.get("story"):
            story.append(res["story"].strip())
    moments.sort(key=lambda m: m["time"])
    return moments, " ".join(story)


# ---------------------------------------------------------------------------
# 2. what the story means to say, against what comes across
# ---------------------------------------------------------------------------
def compare_pass(st, a, r, moments, viewer_story):
    import ss_claude as C
    mapping = _mapping(a)
    seen = "\n".join(f"{fmt_time(m['time'])}-{fmt_time(m['end'])} ({m['time']:.1f}s) "
                     f"[{m['clear']}] {m['happens']}"
                     + (f"  <- wondering: {m['wondering']}" if m["wondering"] else "")
                     for m in moments)
    heard = "\n".join(f"{fmt_time(l['s'])}: {l['text']}" for l in a.get("lines", []))
    prompt = f"""A {kind_text(a['kind'])} called "{a['title']}", {a['duration']:.0f} seconds long.

WHAT THE STORY MEANS TO SAY (her script):
{(a.get('story_text') or '').strip()[:16000] or '(no script given: use the shot list below)'}
{f"SUMMARY: {r['summary']}" if r and r.get('summary') else ''}

THE SHOTS ON THE TIMELINE, IN ORDER (what each is meant to show, and the prompts it was made from):
{_shot_listing(a['shots'], r, mapping)}

CHARACTERS (describe them in prompts by these looks, never by name):
{_cast(r)}

THE SOUNDS ON THE TIMELINE:
{_sound_listing(a.get('sounds', []))}

{"WHAT IS SAID OR SUNG:" + chr(10) + heard if heard else ""}

WHAT A FIRST-TIME VIEWER UNDERSTOOD (they watched without the script; [clear], [unsure] or \
[lost], and what they were wondering):
{seen}

THE STORY AS THAT VIEWER WOULD TELL IT: {viewer_story}

Compare what the story means to say with what came across. Write a note for every place where \
a viewer would miss, misread or not feel something that matters: a character's feeling or \
goal that doesn't show, an action that reads as something else, a jump the viewer can't \
follow, a joke without its set-up or without a reaction, a moment too quick to take in, a \
missing look at what a character sees, an unclear place or time, a payoff or ending that \
doesn't land. Give the times from the viewer's notes. Don't note what already works (put the \
best of that in "working"). Usually 5 to 15 notes for a few minutes; fewer if it reads well.

For every note give 1 to 3 fixes, the easiest that will really work first. Shot numbers are as \
listed above (\"6\", \"20.1\"). For redo_clip, keep what the shot is for and write a new \
image-to-video prompt that makes the action read clearly (and a new text-to-image prompt only if \
the starting picture itself is the problem). For new_shot, say which shot it comes after and \
write both prompts. For sound, write the sound prompt and give the times."""
    res = C.ask(prompt, compare_schema(), system=_advisor_system(), timeout=1200)
    labels = _labels(a, r)
    notes = []
    for n in res.get("notes", []):
        try:
            t = max(0.0, float(n.get("time") or 0))
            e = max(t, float(n.get("end") or t))
        except (TypeError, ValueError):
            continue
        fixes = [_clean_fix(f, labels) for f in (n.get("fixes") or [])][:3]
        notes.append({"time": round(t, 2), "end": round(e, 2),
                      "severity": n.get("severity") if n.get("severity") in ("fix", "suggest")
                      else "suggest", "kind": n.get("kind", "unclear"),
                      "title": (n.get("title") or "Note").strip(),
                      "problem": (n.get("problem") or "").strip(), "fixes": fixes})
    notes.sort(key=lambda n: n["time"])
    summary = {"comes_across": res.get("comes_across", "mostly"),
               "overall": (res.get("overall") or "").strip(),
               "working": [x.strip() for x in res.get("working") or [] if x.strip()][:4],
               "top_fixes": [x.strip() for x in res.get("top_fixes") or [] if x.strip()][:3]}
    return notes, summary


def audit(st, settings=None, progress=None, cancel=None):
    import storysound as E
    import ss_claude as C
    say = progress or (lambda f, m: None)
    if not C.provider():
        raise StoryError("The story check needs Claude or ChatGPT. Open Settings (the gear at "
                         "the top) to set up ChatGPT.")
    a = prepare(st, lambda f, m: say(0.35 * f, m), cancel)
    if cancel and cancel():
        raise StoryError("Stopped.")
    r = _shot_list(a)
    work = Path(st["work"]) / "story"
    dur = a["duration"]
    every = 1.0 if dur <= 420 else 2.0 if dur <= 900 else 3.0
    say(0.36, "Taking a picture of every second…")
    frames = M.stills(st["picture"], work, every=every)
    sheets = M.contact_sheets(frames, work, label="story")
    moments, viewer_story = viewer_pass(st, a, sheets, every,
                                        lambda f, m: say(0.38 + 0.3 * f, m), cancel)
    if cancel and cancel():
        raise StoryError("Stopped.")
    say(0.7, f"{C.ai_name()} is comparing what came across with your story…")
    notes, summary = compare_pass(st, a, r, moments, viewer_story)
    for i, n in enumerate(notes):
        n["id"] = i + 1
    a.update({"moments": moments, "viewer_story": viewer_story, "notes": notes,
              "summary": summary, "by": C.ai_name(), "made": time.strftime("%Y-%m-%d %H:%M"),
              "checked": True})
    st["stage"] = "audit"
    say(0.93, "Putting the notes on your timeline…")
    write_outputs(st)
    E.save(st)
    log("story check:", len(moments), "moments,", len(notes), "notes, comes across:",
        summary["comes_across"], "| shot list:", a.get("shot_list") or "none")
    say(1.0, f"Done: {len(notes)} notes.")
    return a


# ---------------------------------------------------------------------------
# 3. "what I'd like better", in her own words
# ---------------------------------------------------------------------------
def ask(st, text, at=None, progress=None, cancel=None):
    import storysound as E
    import ss_claude as C
    say = progress or (lambda f, m: None)
    text = (text or "").strip()
    if not text:
        raise StoryError("Type what you'd like better first.")
    if not C.provider():
        raise StoryError("This needs Claude or ChatGPT. Open Settings (the gear at the top) to "
                         "set up ChatGPT.")
    a = st.get("audit") or {}
    if not a.get("shots") or not Path(st.get("picture") or "").exists():
        a = prepare(st, lambda f, m: say(0.5 * f, m), cancel)
    r = _shot_list(a)
    dur = a["duration"]
    work = Path(st["work"]) / "story"
    try:
        at = None if at is None or at == "" else min(max(0.0, float(at)), dur)
    except (TypeError, ValueError):
        at = None
    say(0.55, "Looking closely at that moment…" if at is not None else "Looking at the video…")
    if at is not None:
        lo, hi = max(0.0, at - 4.0), min(dur, at + 5.0)
        frames = M.burst(st["picture"], lo, hi, 24, work, f"ask{len(a['asks']) + 1:03d}")
        sheets = M.contact_sheets(frames, work, per_sheet=12, cols=4,
                                  label=f"ask{len(a['asks']) + 1:03d}_",
                                  fmt=lambda t: f"{t:.1f}s")
        where = (f"She is pointing at {fmt_time(at)} ({at:.1f}s). The stills cover {lo:.1f}s to "
                 f"{hi:.1f}s, about {(hi - lo) / 24:.2f}s apart.")
    else:
        lo, hi = 0.0, dur
        every = max(1.0, round(dur / 100.0))
        frames = M.stills(st["picture"], work, every=every)
        sheets = M.contact_sheets(frames, work, label="askall")
        where = f"She didn't point at a time. The stills cover the whole video, one every {every:g} s."
    mapping = _mapping(a)
    near_lo, near_hi = (lo - 6, hi + 6) if at is not None else (None, None)
    before = "\n".join(f"SHE SAID: {x['text']}\nYOU ANSWERED: {x.get('reply', '')}"
                       for x in a["asks"][-4:])
    seen = "\n".join(f"{fmt_time(m['time'])} [{m['clear']}] {m['happens']}"
                     for m in a.get("moments", [])
                     if near_lo is None or near_lo <= m["time"] <= near_hi)
    heard = "\n".join(f"{fmt_time(l['s'])}: {l['text']}" for l in a.get("lines", [])
                      if near_lo is None or (l["e"] >= near_lo and l["s"] <= near_hi))
    prompt = f"""A {kind_text(a['kind'])} called "{a['title']}", {dur:.0f} seconds long.
WHAT THE STORY MEANS TO SAY: {((r or {}).get('summary') or '') or (a.get('story_text') or '')[:3000] or '(not given)'}

{where}

THE SHOTS {'AROUND THAT MOMENT' if at is not None else 'IN ORDER'}:
{_shot_listing(a['shots'], r, mapping, near_lo, near_hi) if at is not None else _shot_listing(a['shots'], r, mapping)}

CHARACTERS (describe them in prompts by these looks, never by name):
{_cast(r)}

SOUNDS ON THE TIMELINE {'THERE' if at is not None else ''}:
{_sound_listing(a.get('sounds', []), near_lo, near_hi)}
{("WORDS HEARD:" + chr(10) + heard) if heard else ""}
{("WHAT A FIRST-TIME VIEWER UNDERSTOOD THERE:" + chr(10) + seen) if seen else ""}
{("EARLIER IN THIS CONVERSATION:" + chr(10) + before) if before else ""}

WHAT SHE SEES AND WOULD LIKE BETTER, IN HER WORDS:
"{text}"

Look at the stills, work out what she means and why it looks that way, and give the fixes \
most likely to work: change the edit, add a sound, make the clip again with a new image-to-video \
prompt (and a new text-to-image prompt if the starting picture is the problem), add a new shot, \
or change the words. Shot numbers are as listed. Times are seconds from the start."""
    images = [(f"Stills {x:.1f}s to {y:.1f}s:" if at is not None else
               f"Stills {fmt_time(x)} to {fmt_time(y)}:", p) for x, y, p in sheets]
    say(0.6, f"{C.ai_name()} is looking at what you said…")
    res = C.ask(prompt, ask_schema(), system=_advisor_system(), images=images, timeout=900)
    labels = _labels(a, r)
    item = {"id": (max([x["id"] for x in a["asks"]] or [0]) + 1), "text": text,
            "at": round(at, 2) if at is not None else None,
            "reply": (res.get("reply") or "").strip(),
            "question": (res.get("question") or "").strip(),
            "fixes": [_clean_fix(f, labels) for f in (res.get("fixes") or [])][:4],
            "when": time.strftime("%Y-%m-%d %H:%M")}
    a["asks"] = (a["asks"] + [item])[-MAX_ASKS:]
    st["audit"] = a
    if st.get("stage") in (None, "opened"):
        st["stage"] = "audit"
    E.save(st)
    log("story check: her note answered with", len(item["fixes"]), "fixes:", repr(text)[:160])
    say(1.0, "Done.")
    return item


# ---------------------------------------------------------------------------
# putting fixes to work
# ---------------------------------------------------------------------------
def find_fix(a, ref, k):
    """ref "n3" = note 3, "a2" = her note 2; k = the fix's place in its list."""
    pool = a.get("notes", []) if str(ref).startswith("n") else a.get("asks", [])
    try:
        item = next(x for x in pool if x["id"] == int(str(ref)[1:]))
        return item, item["fixes"][int(k)]
    except (StopIteration, ValueError, IndexError, KeyError):
        raise StoryError("That fix isn't there any more. Run the story check again.")


def _work_of(project, seq_name):
    import storysound as E
    h = hashlib.md5(f"{Path(project).resolve()}|{seq_name}".encode()).hexdigest()[:8]
    return E.WORK / f"{safe_name(Path(project).stem)[:40]}-{h}"


def _sources(project):
    """This project, then the projects it was copied from: "X (with sound, audio fixed)" ->
    "X (with sound)" -> "X"."""
    import storysound as E
    p = Path(project)
    out = [p]
    m = re.search(r" \(([^()]*)\)$", p.stem)
    if m:
        tags = [re.sub(r" \d+$", "", t.strip()) for t in m.group(1).split(",")]
        if tags and all(t in E.COPY_TAGS for t in tags):
            base = p.stem[:m.start()]
            for i in range(len(tags) - 1, -1, -1):
                out.append(p.with_name(f"{base} ({', '.join(tags[:i])}).prproj" if i else
                                       f"{base}.prproj"))
    return out


def sound_home(st):
    """Where this movie's StoryStudio sound list lives: (project, state file), or (None, None)."""
    for proj in _sources(st["project"]):
        if not proj.exists():
            continue
        work = _work_of(proj, st["seq_name"])
        for name in ("state.json", "state_sounds.json"):
            f = work / name
            try:
                data = json.loads(f.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                continue
            if data.get("cues"):
                return str(proj), f
    return None, None


def sound_source(st):
    """The StoryStudio sounds this timeline was given and the timeline they were placed on:
    (state with "cues", project the sounds were made for). Found by name first, then by the
    sequence itself (a project saved under a new name keeps its sequence)."""
    import storysound as E
    proj, f = sound_home(st)
    if proj:
        return json.loads(Path(f).read_text(encoding="utf-8-sig")), proj
    best = None
    for f in list(E.WORK.glob("*/state.json")) + list(E.WORK.glob("*/state_sounds.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if data.get("seq_uid") != st["seq_uid"] or not data.get("cues"):
            continue
        made = [c for c in data["cues"] if c.get("file") and c.get("keep", True)]
        if made and (best is None or f.stat().st_mtime > best[0]):
            best = (f.stat().st_mtime, data)
    return (best[1], best[1]["project"]) if best else (None, None)


def line_up(st):
    """After shots were moved, trimmed or deleted in Premiere: a copy of this project with
    every sound effect back on its shot."""
    import storysound as E
    import ss_prwrite as W
    data, proj = sound_source(st)
    if not data:
        raise StoryError("StoryStudio couldn't find the sound effects it made for this video. "
                         "Open the project you added sound effects to (or a copy of it).")
    ref = data.get("snapshot") if Path(data.get("snapshot") or "").exists() else proj
    if not Path(ref or "").exists():
        raise StoryError("StoryStudio's copy of the timeline the sounds were made for is gone, "
                         "so it can't tell where the shots were.")
    sounds = [{"name": c["what"], "start": float(c["place_at"]),
               "end": float(c["place_at"]) + float(c.get("length") or (c["end"] - c["start"]))}
              for c in data["cues"] if c.get("file") and c.get("keep", True) and
              c.get("kind") not in E.BEDS]
    real = Path(st["project"])
    out = E.copy_path(real, "sounds lined up")
    res = W.line_up_sounds(st.get("snapshot") or real, st["seq_uid"], ref, out,
                           ref_sounds=sounds)
    log("lined up sounds from", Path(proj).name, res)
    return res


def add_sound(st, ref, k):
    """Put a sound fix into the movie's sound list (not made yet): it is made with "Make the
    missing" or "Use free sound effects" on the sounds page."""
    import storysound as E
    a = st.get("audit") or {}
    item, fx = find_fix(a, ref, k)
    if fx["type"] != "sound":
        raise StoryError("That fix isn't a sound.")
    proj, f = sound_home(st)
    if not proj:
        raise StoryError("This movie doesn't have StoryStudio sound effects yet. Add sound "
                         "effects first (on the project page); then this sound can go in with "
                         "them.")
    same = Path(f).parent == Path(st["work"]) and st.get("cues")
    data = st if same else json.loads(Path(f).read_text(encoding="utf-8-sig"))
    cues = data["cues"]
    for c in cues:                                 # already added
        if c.get("from_story") == f"{ref}:{k}":
            return {"project": proj, "already": True, "what": c["what"]}
    used = {c["id"] for c in cues}
    i = len(cues) + 1
    while f"c{i:03d}" in used:
        i += 1
    kind = SOUND_KIND.get(fx.get("sound_kind") or "effect", "action")
    start, end = float(fx["at"]), max(float(fx["end"]), float(fx["at"]) + 0.4)
    if kind == "music" and end - start < 6:        # a ding or a little fanfare is an effect,
        kind = "action"                            # not a background tune made as a loop
    words = re.sub(r"[^a-z ]", " ", fx["title"].lower()).split()
    cue = {"id": f"c{i:03d}", "kind": kind, "start": round(start, 3), "end": round(end, 3),
           "what": fx["title"], "sound_prompt": fx["sound"], "adobe_prompt": fx["sound"],
           "suno_prompt": fx["sound"] if kind == "music" else "",
           "search_words": " ".join(words[:3]), "synced": kind != "music",
           "level": "background" if kind == "music" else "medium", "includes": [],
           "keep": True, "source": "", "file": "", "place_at": round(start, 3),
           "length": round(end - start, 3), "note": "Added from the story check: " + fx["what"],
           "engine": "", "feeling": "", "added_by_check": True, "from_story": f"{ref}:{k}"}
    cues.append(cue)
    cues.sort(key=lambda c: (c["start"], c["end"]))
    if same:
        E.save(st)
    else:
        Path(f).write_text(json.dumps(data, indent=1), encoding="utf8")
    fx["added"] = proj
    if not same:
        E.save(st)                                 # remembers that this fix went in
    log("story check: sound added to", Path(f).parent.name, cue["id"], cue["what"])
    return {"project": proj, "already": False, "what": cue["what"]}


def to_shot_list(st, ref, k):
    """Save a "make the clip again" or "new shot" fix into the shot list: the shot gets the new
    prompts and is marked to make again; a new shot goes in after the one it follows, numbered
    like "20.1" (the way she names extra shots)."""
    import storysound as E
    import ss_script as SB
    if str(ref).startswith("f"):                   # a "make again" note on the feedback page
        a = st.get("feedback") or {}
        item, fx = find_fb_fix(a, ref, k)
    else:
        a = st.get("audit") or {}
        item, fx = find_fix(a, ref, k)
    if fx["type"] not in ("redo_clip", "new_shot"):
        raise StoryError("Only new clips and new shots go into the shot list.")
    if not a.get("shot_list"):
        raise StoryError("StoryStudio didn't find the shot list this video was made from, so "
                         "copy the prompts instead.")
    r = SB.load(a["shot_list"])
    style = r.get("style") or {}
    if not style.get("style") and style.get("id"):
        style = SB.find_style(style["id"]) or style

    def pic_prompt(s):
        try:
            if style.get("style"):
                return SB.picture_prompt(s, r, style, r.get("aspect", "16:9"))
        except Exception:
            log_error("story check picture prompt")
        return s["picture"]
    by_label = {shot_label(s): s for s in r["shots"]}
    base = by_label.get(fx["shot"])
    if fx["type"] == "redo_clip":
        if not base:
            raise StoryError("StoryStudio couldn't tell which shot this is, so copy the prompts "
                             "instead.")
        base.setdefault("before_story_check", {"picture": base.get("picture", ""),
                                               "motion": base.get("motion", "")})
        if fx["picture"]:
            base["picture"] = fx["picture"]
            base["picture_prompt"] = pic_prompt(base)
        base["motion"] = fx["motion"] or base.get("motion", "")
        base["camera"], base["camera2"] = fx["camera"] or "Static shot", ""
        base.pop("camera_asked", None)
        base.pop("camera2_asked", None)
        base["minimax_prompt"] = SB.minimax_prompt(base)
        base["status"] = "redo picture" if fx["picture"] else "redo clip"
        base["story_note"] = fx["what"]
        shot = base
    else:
        after = base or (r["shots"][-1] if r["shots"] else None)
        root = re.match(r"^\d+", shot_label(after)).group(0) if after and \
            re.match(r"^\d+", shot_label(after)) else str(len(r["shots"]))
        k2 = 1
        while f"{root}.{k2}" in by_label:
            k2 += 1
        n = max([s["n"] for s in r["shots"]] or [0]) + 1
        t = float(after["end"]) if after else 0.0
        shot = {"n": n, "label": f"{root}.{k2}", "scene": (after or {}).get("scene", 1),
                "title": fx["title"], "purpose": fx["what"], "words": "",
                "picture": fx["picture"] or fx["what"], "motion": fx["motion"],
                "camera": fx["camera"] or "Static shot", "camera2": "",
                "characters": [c["name"] for c in r.get("characters", [])
                               if c["name"].lower() in (fx["picture"] + fx["motion"]).lower()],
                "extra": True, "start": t, "end": t, "paths": [], "files": [],
                "status": "todo", "story_note": fx["what"], "added_by_story_check": True}
        shot["picture_prompt"] = pic_prompt(shot)
        shot["minimax_prompt"] = SB.minimax_prompt(shot)
        i = r["shots"].index(after) + 1 if after else len(r["shots"])
        r["shots"].insert(i, shot)
    r["files"] = SB.save(r)
    fx["saved"] = shot_label(shot)
    E.save(st)
    log("story check: shot list", fx["type"], "shot", shot_label(shot))
    return {"label": shot_label(shot), "path": r["files"]["json"], "new": fx["type"] == "new_shot"}


# ---------------------------------------------------------------------------
# "make again" notes from Edit feedback: full prompts, ready for the shot list
# ---------------------------------------------------------------------------
REDO_KINDS = ("ai_glitch", "continuity")

REDO_SYSTEM = """You are an experienced director helping a storyteller who makes children's \
stories, silent cartoons and music videos with AI pictures (ChatGPT or Nano Banana), AI video \
clips (MiniMax Hailuo, image to video) and Adobe Premiere. She is not a professional: write in \
plain, warm, everyday words.

An editor watched her video and found clips that look wrong: AI glitches (bodies that stretch or \
melt, extra tails, things that morph) or things that change between shots (a character's colour \
or shape, a prop's size, the season). For each one, look at the stills and the shot's old \
prompts and write what she needs to make that clip again so it comes out right:
- redo_clip: the shot made again. Always write the full new image-to-video prompt: the same \
action and story purpose as before, described so the glitch can't happen (simple, steady \
movement; one clear action; the character's look stated plainly). When the problem is already \
in the starting picture (wrong colour, wrong size, wrong season, a missing cap), also write a \
full new text-to-image prompt; otherwise leave the picture empty.
- edit: only when making it again isn't worth it because a trim or a cut hides the problem \
completely. Say exactly what to trim or cut.
Use the shot numbers exactly as listed. Describe characters by their looks, never by name.
Many of these problems ARE things changing between shots, so the new prompts must all match: \
describe each character, the main props and the place with the words given under CHARACTERS \
AND HOW THINGS LOOK, the same words in every prompt.

Prompt rules:
TEXT-TO-IMAGE: {picture}
IMAGE-TO-VIDEO: {motion}

Studio rules: no religious symbols or churches; Jesus or God only ever as a warm light; no \
Christmas, Halloween, Easter or St. Patrick's Day imagery."""


LOOKS_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {"type": "array", "items": {
            "type": "object",
            "properties": {"who": {"type": "string", "description": "a short plain label, e.g. "
                                   "'the squirrel'"},
                           "look": {"type": "string", "description": "one fixed description "
                                    "for prompts, 12-30 words: kind, size, colours, face, "
                                    "clothes, anything that makes them recognisable"}},
            "required": ["who", "look"], "additionalProperties": False}},
        "props": {"type": "string", "description": "the main objects the story is about, "
                  "described the same fixed way (size compared with the character, colours, "
                  "shape, parts), one sentence each; empty if none"},
        "place": {"type": "string", "description": "the place, season, time of day and light, "
                  "as one fixed description"},
    },
    "required": ["characters", "props", "place"],
    "additionalProperties": False,
}


def _hint(fb):
    """The editor's own advice on keeping things the same, if it gave one."""
    for t in (fb.get("summary") or {}).get("top_fixes") or []:
        m = re.search(r"(?:description|describe)[^:]{0,60}:\s*(.+)", t, flags=re.I)
        if m:
            return m.group(1).strip()
    return ""


def fixed_looks(st, picture, dur, r, work, say):
    """One fixed description of the characters, props and place, taken from how they look in
    MOST of the shots, so every new prompt matches the rest of the movie."""
    import ss_claude as C
    fb = st["feedback"]
    if fb.get("looks"):
        return fb["looks"]
    every = max(2.0, round(dur / 30.0))
    frames = M.stills(picture, work, every=every)
    sheets = M.contact_sheets(frames, work, per_sheet=16, cols=4, label="looks_")
    cast = _cast(r)
    prompt = f"""Stills from the whole of a {kind_text(fb.get('kind') or 'story')} called \
"{st['seq_name']}", one every {every:g} seconds. The clips were made by AI, so some shots show \
a character or prop looking different (another colour, size or shape). Write ONE fixed \
description of each recurring character, the main props and the place, going by how they look \
in MOST of the shots, so that every clip made again matches the rest of the movie. \
{('Characters listed in the shot list:' + chr(10) + cast) if r and r.get('characters') else ''}
{('The editor suggested: ' + _hint(fb)) if _hint(fb) else ''}"""
    images = [(f"Stills {fmt_time(x)} to {fmt_time(y)}:", p) for x, y, p in sheets]
    res = C.ask(prompt, LOOKS_SCHEMA, images=images, timeout=600)
    lines = [f"- {c['who']}: {c['look']}" for c in res.get("characters") or []]
    if res.get("props"):
        lines.append(f"- props: {res['props']}")
    if res.get("place"):
        lines.append(f"- place: {res['place']}")
    fb["looks"] = "\n".join(lines)
    log("feedback: fixed looks", fb["looks"].replace("\n", " | ")[:400])
    return fb["looks"]


def redo_notes(fb):
    return [n for n in fb.get("notes", []) if n.get("kind") in REDO_KINDS and
            n.get("severity") != "good" and not (n.get("edit") and fb.get("kind") == "cartoon")]


def redo_schema():
    item = fix_schema()
    item["properties"]["type"] = {"type": "string", "enum": ["redo_clip", "edit"]}
    item["properties"]["note"] = {"type": "integer", "description": "the note's number"}
    item["required"] = ["note"] + item["required"]
    return {"type": "object", "properties": {"fixes": {"type": "array", "items": item}},
            "required": ["fixes"], "additionalProperties": False}


def find_fb_fix(fb, ref, k):
    try:
        item = next(x for x in fb.get("notes", []) if x["id"] == int(str(ref)[1:]))
        return item, item["fixes"][int(k)]
    except (StopIteration, ValueError, IndexError, KeyError):
        raise StoryError("That prompt isn't there any more. Write the prompts again.")


def redo_prompts(st, progress=None, cancel=None, per_request=5, workers=3):
    """Write the new prompts for every "make again" note on the feedback page."""
    import storysound as E
    import ss_claude as C
    import ss_script as SB
    import threading
    say = progress or (lambda f, m: None)
    fb = st.get("feedback") or {}
    notes = redo_notes(fb)
    if not notes:
        raise StoryError("There are no clips to make again in these notes.")
    if not C.provider():
        raise StoryError("This needs Claude or ChatGPT. Open Settings (the gear at the top) to "
                         "set up ChatGPT.")
    say(0.02, "Finding your shot list…")
    seq = E._sequence(st)
    dur = seq["duration"]
    tl = timeline_shots(seq)
    path, r, mapping = find_shot_list(tl)
    labels = _labels({}, r)
    picture = st.get("picture") or ""
    if not Path(picture).exists():
        say(0.04, "Getting the picture ready…")
        picture = st["picture"] = M.build_picture(seq, Path(st["work"]), lambda f: say(
            0.04 + 0.1 * f, "Getting the picture ready…"))
    work = Path(st["work"]) / "redo"
    work.mkdir(parents=True, exist_ok=True)
    say(0.15, "Taking a close look at each clip…")
    looks = []
    for i, n in enumerate(notes):
        if cancel and cancel():
            raise StoryError("Stopped.")
        lo = max(0.0, float(n["time"]) - 0.5)
        hi = min(dur, max(lo + 2.0, min(float(n.get("end") or n["time"]), lo + 8.0)) + 0.5)
        frames = M.burst(picture, lo, hi, 8, work, f"n{n['id']:03d}")
        sheet = M.contact_sheets(frames, work, per_sheet=8, cols=4, label=f"n{n['id']:03d}_",
                                 fmt=lambda t: f"{t:.1f}s")
        looks.append((n, lo, hi, sheet[0][2] if sheet else None))
        say(0.15 + 0.2 * (i + 1) / len(notes), "Taking a close look at each clip…")
    say(0.36, "Working out how the characters and everything else should look…")
    try:
        look_text = fixed_looks(st, picture, dur, r, work, say)
    except StoryError:
        raise
    except Exception:
        log_error("feedback fixed looks")
        look_text = ""
    system = REDO_SYSTEM.format(picture=SB.PICTURE_RULES, motion=SB.MOTION_RULES)
    kind = kind_text(fb.get("kind") or (r or {}).get("kind") or "story")
    head = f"""A {kind} called "{st['seq_name']}".
WHAT THE STORY IS ABOUT: {((r or {}).get('summary') or '')[:1500] or '(not given)'}
CHARACTERS AND HOW THINGS LOOK (use these exact words in every prompt; never use names):
{look_text or _cast(r)}
"""
    groups =[looks[i:i + per_request] for i in range(0, len(looks), per_request)]
    got, lock, done = {}, threading.Lock(), [0]

    def one(group):
        parts, images = [], []
        for n, lo, hi, sheet in group:
            parts.append(f"""NOTE {n['id']} at {fmt_time(n['time'])} ({lo:.1f}s to {hi:.1f}s): {n['title']}
What the editor saw: {n['note']}
The editor's suggestion: {n.get('how') or '(none)'}
The shots there:
{_shot_listing(tl, r, mapping, lo, hi)}""")
            if sheet:
                images.append((f"Note {n['id']}: stills {lo:.1f}s to {hi:.1f}s", sheet))
        prompt = head + "\n" + "\n\n".join(parts) + """

For each note, give one fix (two only if the problem really covers two different shots). \
Each fix says which note it is for. Times are seconds from the start of the movie."""
        res = C.ask(prompt, redo_schema(), system=system, images=images, timeout=900)
        with lock:
            for f in res.get("fixes") or []:
                try:
                    got.setdefault(int(f.get("note")), []).append(f)
                except (TypeError, ValueError):
                    pass
            done[0] += 1
            say(0.42 + 0.55 * done[0] / len(groups), f"{C.ai_name()} is writing the prompts…")

    say(0.42, f"{C.ai_name()} is writing the prompts…")
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in [ex.submit(one, g) for g in groups]:
            try:
                fut.result()
            except Exception as e:
                errors.append(e)
                log_error("feedback redo prompts")
    if errors and not got:
        raise errors[0] if isinstance(errors[0], StoryError) else \
            StoryError("The prompts couldn't be written this time. Try again in a minute.")
    count = 0
    for n in notes:
        fixes = [_clean_fix(dict(f, type=f.get("type") or "redo_clip"), labels)
                 for f in got.get(n["id"], [])][:2]
        if fixes:
            n["fixes"] = fixes
            count += 1
    fb["shot_list"] = path or ""
    fb["redo_written"] = time.strftime("%Y-%m-%d %H:%M")
    st["feedback"] = fb
    E.save(st)
    log("feedback: prompts for", count, "of", len(notes), "clips to make again | shot list:",
        path or "none")
    say(1.0, "Done.")
    return {"written": count, "notes": len(notes)}


# ---------------------------------------------------------------------------
# the notes on the timeline and in a list
# ---------------------------------------------------------------------------
COLOUR = {"fix": "red", "suggest": "yellow"}


def write_outputs(st):
    import storysound as E
    import ss_prwrite as W
    a = st["audit"]
    od = E.out_dir(st)
    name = safe_name(st["seq_name"])
    out = {"folder": str(od)}
    lst = od / f"{name} - story check.txt"
    lst.write_text(notes_text(st), encoding="utf8")
    out["list"] = str(lst)
    proj = E.copy_path(Path(st["project"]), "story check")
    markers = []
    for n in a.get("notes", []):
        how = "\n".join(f"- {FIX_NAMES[f['type']]}: {f['what']}" for f in n["fixes"])
        markers.append({"start": n["time"], "end": n["end"] if n["end"] > n["time"] + 0.3 else None,
                        "colour": COLOUR.get(n["severity"], "yellow"),
                        "name": f"Story: {n['title']}",
                        "comment": n["problem"] + (f"\n\n{how}" if how else "")})
    try:
        out["project"] = W.write_markers(st.get("snapshot") or st["project"], st["seq_uid"],
                                         markers, proj)
    except StoryError as ex:
        out["project_error"] = str(ex)
    except Exception:
        log_error("story check markers")
        out["project_error"] = ("StoryStudio couldn't put the notes on the timeline, but the "
                                "notes list is in the StoryStudio folder.")
    a["outputs"] = out
    return out


def notes_text(st):
    a = st["audit"]
    s = a.get("summary") or {}
    L = [f"Story check for \"{st['seq_name']}\"", ""]
    if s.get("overall"):
        L += [s["overall"], ""]
    if a.get("viewer_story"):
        L += ["The story as a first-time viewer would tell it:", "  " + a["viewer_story"], ""]
    if s.get("working"):
        L += ["What tells the story well:"] + [f"  + {x}" for x in s["working"]] + [""]
    if s.get("top_fixes"):
        L += ["What would help the story most:"] + [f"  - {x}" for x in s["top_fixes"]] + [""]
    for n in a.get("notes", []):
        L.append(f"{fmt_time(n['time'])}  {'PLEASE FIX' if n['severity'] == 'fix' else 'Idea'}: "
                 f"{n['title']}")
        L.append(f"  {n['problem']}")
        for f in n["fixes"]:
            L.append(f"  * {FIX_NAMES[f['type']]}{' (shot ' + f['shot'] + ')' if f['shot'] else ''}: "
                     f"{f['what']}")
            if f.get("sound"):
                L.append(f"      Sound: {f['sound']}")
            if f.get("picture"):
                L.append(f"      Text-to-image prompt: {f['picture']}")
            if f.get("motion"):
                L.append(f"      Image-to-video prompt: {f['motion']} [{f['camera']}]")
        L.append("")
    return "\n".join(L)
