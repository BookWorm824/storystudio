"""Starting from clips that are already made: put them in order against the script.

    folder_view(folder)               -> what is in the folder (for the setup page)
    build(spec, progress, cancel)     -> a shot list (saved), with a rough cut written

The user has a folder of clips named with their shot numbers ("Shot 01.mp4", and extra shots
as "Shot 20.1.mp4" or "Shot 48b.mp4"), the script (the story as it is narrated, song lyrics,
or a cartoon outline) and maybe the narration or song as a sound file. StoryStudio looks at a
still from every clip, works out which words of the script belong with each one, listens to
the recording to find when those words are said, and writes a Premiere timeline with every
clip placed where its words are heard.

Everything is optional except the folder: with no script the clips simply go in number
order; with no recording they go back to back; with no AI they go in number order too.
"""
import difflib
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ss_core import APP_DATA, StoryError, log, log_error, probe
import ss_media as M
import ss_clips as K
import ss_voices as V

SOUNDS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".aif", ".aiff"}
PER_REQUEST = 60                # shots shown to the AI at a time (five sheets of twelve)
LEAD = 0.25                     # a cut lands this long before the first word of its shot
WORD = 0.32                     # a spoken word, roughly, in seconds
SAY = 0.4                       # a narrated word with its share of the pauses
GOOD_MATCH = 0.45               # the share of script words that must be heard to trust times

SUB = r"(?:[.\-_](\d{1,2})(?!\d)|([a-z])(?![a-z]))?"
NAMED = re.compile(r"(?:shot|sh|clip|scene)[\s_\-#]*0*(\d{1,3})" + SUB, re.I)
LEADING = re.compile(r"^\s*0*(\d{1,3})" + SUB, re.I)
COPY = re.compile(r"\((?:alt|copy)?\s*\d+\)|\bcopy\b|\balt\s*\d+\b", re.I)


def label_of(name):
    """The shot a file is, as the user numbered it, with a key to sort by:
    "Shot 20.1.mp4" -> ("20.1", (20, 1)); "Shot 48b.mp4" -> ("48b", (48, 2));
    "07 - splash.png" -> ("7", (7, 0)); None when the name has no shot number."""
    stem = Path(name).stem
    m = NAMED.search(stem) or LEADING.match(stem)
    if not m:
        return None
    n = int(m.group(1))
    if m.group(2):
        return f"{n}.{int(m.group(2))}", (n, int(m.group(2)))
    if m.group(3):
        return f"{n}{m.group(3).lower()}", (n, ord(m.group(3).lower()) - 96)
    return str(n), (n, 0)


def look(folder):
    """Every shot in the folder, in order: [{"label", "key", "clip", "picture"}], the names
    left out (no shot number), how many copies were skipped, and any sound files lying in
    the folder (one of them is probably the narration or the song)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise StoryError("That folder couldn't be opened.")
    places = [folder]
    for sub in folder.iterdir():
        if sub.is_dir() and sub.name.lower() != "storystudio":
            places.append(sub)
    by, unknown, sounds, copies = {}, [], [], 0
    for d in places:
        try:
            files = [p for p in d.iterdir() if p.is_file()]
        except OSError:
            continue
        for p in sorted(files, key=lambda p: p.name.lower()):
            ext = p.suffix.lower()
            if ext in SOUNDS:
                sounds.append(str(p))
                continue
            if ext not in K.PICTURES and ext not in K.CLIPS:
                continue
            if "(karaoke)" in p.name.lower() or "(lyrics)" in p.name.lower() or \
                "storystudio" in p.name.lower():
                continue
            got = label_of(p.name)
            if not got:
                unknown.append(p.name)
                continue
            kind = "clip" if ext in K.CLIPS else "picture"
            d2 = by.setdefault(got[0], {"label": got[0], "key": got[1], "clip": [],
                                        "picture": []})
            d2[kind].append((bool(COPY.search(p.stem)), -p.stat().st_mtime, str(p)))
    shots = []
    for d2 in sorted(by.values(), key=lambda x: x["key"]):
        for kind in ("clip", "picture"):                   # the original before any copy
            got = sorted(d2[kind])
            copies += max(0, len(got) - 1)
            d2[kind] = got[0][2] if got else ""
        shots.append(d2)
    return shots, unknown, copies, sorted(sounds, key=lambda p: -Path(p).stat().st_size)


def voices_view(folder):
    """What the setup page shows about a folder of voice files."""
    import ss_voices as V
    files = V.voice_files(folder)
    return {"path": str(folder), "name": Path(folder).name, "count": len(files),
            "some": [p.name for p in files[:3]]}


def folder_view(folder):
    """What the setup page shows after a folder is chosen."""
    shots, unknown, copies, sounds = look(folder)
    groups = {}
    for p in sounds:
        groups.setdefault(str(Path(p).parent), []).append(p)
    voice_folders = [voices_view(d) for d, v in groups.items() if len(v) >= 3]
    return {"folder": str(folder), "title": Path(folder).name, "shots": len(shots),
            "first": shots[0]["label"] if shots else "",
            "last": shots[-1]["label"] if shots else "",
            "clips": sum(1 for s in shots if s["clip"]),
            "no_clip": [s["label"] for s in shots if not s["clip"]][:30],
            "left_out": len(unknown), "copies": copies,
            "voice_folders": [v for v in voice_folders if v["count"] >= 3][:3],
            "sounds": [{"path": p, "name": Path(p).name} for p in sounds[:6]]}


# ---------------------------------------------------------------------------
# which words go with which clip
# ---------------------------------------------------------------------------
MATCH = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "one plain sentence: what it is about"},
        "shots": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "shot": {"type": "string", "description": "the shot label printed on the "
                         "still, exactly: \"7\", \"20.1\", \"48b\""},
                "title": {"type": "string", "description": "two to four words naming the shot"},
                "sees": {"type": "string", "description": "one plain sentence: what is on "
                         "screen in this clip"},
                "action": {"type": "string", "description": "what the script says HAPPENS in "
                           "this shot (the movement and the acting), in the script's own "
                           "words, kept to one or two sentences; empty when the script "
                           "doesn't describe the shot's action"},
                "words": {"type": "string", "description": "the exact words of the script "
                          "that are heard while this clip is on screen, word for word; empty "
                          "when nothing is said over it. For a silent cartoon: the shot's "
                          "sound effects as the script lists them"},
                "start": {"type": "number", "description": "when the script prints a time "
                          "for this shot (like 00:05-00:10): where it starts, in seconds "
                          "from the start of the film; 0 when the script gives no time"},
                "end": {"type": "number", "description": "where the script says it ends, in "
                        "seconds from the start of the film; 0 when the script gives no time"},
            },
            "required": ["shot", "title", "sees", "action", "words", "start", "end"],
            "additionalProperties": False}},
    },
    "required": ["summary", "shots"], "additionalProperties": False,
}
SYSTEM = """A storyteller has made every video clip for her short film and has the script. \
She is not a professional. You see one still from each clip, in the order they play, with \
its shot label printed on it (labels like 20.1 or 48b are extra shots that follow shot 20 or \
48). Work out which words of the script are heard while each clip is on screen, so the clips \
can be put on a timeline in the right places.

Rules:
- Use the script's own spoken words exactly, word for word and in order. Never reword, \
shorten or add to them.
- Every spoken word belongs to exactly one shot: nothing left out, nothing used twice, and \
the words run on from one shot to the next in script order.
- The clips are in story order, so each shot's words come after the words of the shot before \
it. A clip was made to show what its words describe: let the picture decide where each \
shot's words begin and end. Cut between shots at the end of a sentence or phrase when the \
pictures allow it, and share the words out so no clip is left holding a very long stretch \
while its neighbours have almost none.
- A clip with nothing said over it (an opening view, a pause, a reaction) gets empty words. \
Don't force words onto it.
- Leave out everything that isn't spoken: shot numbers and headings, picture and video \
prompts, camera notes, speaker labels, stage directions, timings.
- If the script is already divided into numbered shots, clips or scenes, follow its \
division: clip 12 of the script is the still labelled shot 12. A still whose label isn't in \
the script (an extra shot like 48b) gets no words and no time; just say what is on screen.
- When the script prints a time for a shot, pass it on as start and end in seconds. Never \
work a time out yourself: no printed time means 0 for both.
- Answer for every shot label you are shown, once each."""
KIND_TEXT = {
    "story": "a narrated story: the spoken words are the narration (and any dialogue that "
             "is read aloud)",
    "music": "a music video: the spoken words are the song's lyrics as they are sung",
    "cartoon": "a silent cartoon with no narration or dialogue. Here \"words\" means the "
               "shot's SOUND EFFECTS, copied as the script lists them for that shot (leave "
               "the music out); empty when the script lists none. Name each shot and say "
               "what is on screen",
}


def _still(f, work, name):
    """One small still for a shot: a third of the way into its clip, or its picture."""
    try:
        if f["kind"] == "clip":
            sec = f.get("seconds") or 1.0
            got = M.burst(f["path"], sec * 0.25, sec * 0.45, 1, work, name, width=480)
            return got[0][1] if got else None
        return K._small(f["path"], work, name, width=480)
    except Exception:
        log_error("placing: still of " + f["name"])
        return None


def match(kind, script, shots, work, say=None, cancel=None):
    """Ask the AI which words go with which shot. Fills title/purpose/words on the shots it
    answers for and returns the one-line summary."""
    import ss_claude as C
    from PIL import Image
    frames = [(float(s["n"]), s["_still"]) for s in shots if s.get("_still")]
    if not frames:
        return ""
    w0, h0 = Image.open(frames[0][1]).size
    for _, p in frames:
        im = Image.open(p)
        if im.size != (w0, h0):
            im.convert("RGB").resize((w0, h0)).save(p, quality=85)
    by_n = {s["n"]: s for s in shots}
    by_label = {s["label"]: s for s in shots}
    summary, last_words = "", ""
    per = -(-len(frames) // -(-len(frames) // PER_REQUEST))      # even groups: 62 -> 31 + 31
    groups = [frames[i:i + per] for i in range(0, len(frames), per)]
    for g, part in enumerate(groups):
        if cancel and cancel():
            raise StoryError("Stopped.")
        if say:
            say(g / len(groups), f"{C.ai_name()} is matching your clips to the script"
                + (f" ({g + 1} of {len(groups)})" if len(groups) > 1 else "") + "…")
        sheets = M.contact_sheets(part, Path(work) / "close", per_sheet=12, cols=4,
                                  label=f"place{g}_",
                                  fmt=lambda n: "Shot " + by_n[int(n)]["label"])
        mine = {by_n[int(n)]["label"] for n, _ in part}
        a, b = by_n[int(part[0][0])]["label"], by_n[int(part[-1][0])]["label"]
        where = (f"These are shots {a} to {b}; the whole film runs from shot "
                 f"{shots[0]['label']} to shot {shots[-1]['label']}.")
        if g > 0:
            where += (" The shots before these have already used the script up to and "
                      f"including the words: \"{last_words}\". Carry on from there.")
        where += (" The script must be finished by the last of these shots."
                  if g == len(groups) - 1 else
                  f" More shots follow these, so stop where shot {b} ends.")
        res = C.ask(f"This is {KIND_TEXT.get(kind, KIND_TEXT['story'])}.\n{where}\n\n"
                    f"THE SCRIPT:\n{script.strip()}\n\nSay which words go with each shot.",
                    MATCH, system=SYSTEM,
                    images=[(f"Shots {by_n[int(x)]['label']} to {by_n[int(y)]['label']}:", p)
                            for x, y, p in sheets], timeout=900)
        summary = summary or res.get("summary", "")
        for x in res.get("shots", []):
            lab = str(x.get("shot", "")).strip().lower().replace("shot", "").strip()
            s = by_label.get(lab)
            if not s or lab not in mine:
                continue
            s["title"] = (x.get("title") or "").strip()
            s["purpose"] = s["picture"] = (x.get("sees") or "").strip()
            s["motion"] = (x.get("action") or "").strip()
            s["words"] = " ".join((x.get("words") or "").split())
            try:
                a_t, b_t = float(x.get("start") or 0), float(x.get("end") or 0)
            except (TypeError, ValueError):
                a_t = b_t = 0.0
            if b_t - a_t >= 0.5:
                s["_plan"] = (a_t, b_t)
            if s["words"]:
                last_words = " ".join(s["words"].split()[-12:])
    return summary.strip()


# ---------------------------------------------------------------------------
# when the words are said
# ---------------------------------------------------------------------------
def _norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def time_shots(shots, heard, total):
    """Give each shot the moment it starts, from when its words are heard in the recording.
    Returns the share of script words that were found (0-1); below GOOD_MATCH the times
    aren't trusted and nothing is changed."""
    script = [(i, k) for i, s in enumerate(shots) for k in V.tokens(s["words"])]
    hw = [(k, w) for w in heard for k in V.tokens(w["t"])]
    if not script or not hw:
        return 0.0
    sm = difflib.SequenceMatcher(None, [k for _, k in script], [k for k, _ in hw],
                                 autojunk=False)
    first, last, matched = {}, {}, 0
    for a, b, size in sm.get_matching_blocks():
        for j in range(size):
            i, w = script[a + j][0], hw[b + j][1]
            if i not in first:
                before = sum(1 for x in script[:a + j] if x[0] == i)   # its words not heard
                first[i] = max(0.0, w["s"] - before * WORD)
            last[i] = w["e"]
            matched += 1
    quality = matched / len(script)
    if quality < GOOD_MATCH:
        return quality
    n = len(shots)
    starts = [first.get(i) for i in range(n)]
    for i in range(1, n):                                  # never earlier than the shot before
        if starts[i] is not None:
            prev = max((x for x in starts[:i] if x is not None), default=0.0)
            starts[i] = max(starts[i], prev)
    starts[0] = 0.0
    i = 1
    while i < n:                                           # shots whose words weren't found
        if starts[i] is not None:
            i += 1
            continue
        j = i
        while j < n and starts[j] is None:
            j += 1
        a_shot = i - 1
        t0 = starts[a_shot]
        t1 = starts[j] if j < n else max(total, t0 + 1.0)
        quiet = last.get(a_shot, t0)                       # where the shot before stops talking
        run = list(range(i, j))
        room = t1 - quiet
        if all(not shots[k]["words"] for k in run) and room >= 1.2 * len(run):
            for x, k in enumerate(run):                    # wordless shots share the pause
                starts[k] = quiet + LEAD + room * x / len(run)
        else:                                              # share the stretch by word count
            weight = [max(3, len(shots[k]["words"].split())) for k in [a_shot] + run]
            acc = 0.0
            for x, k in enumerate(run):
                acc += weight[x]
                starts[k] = t0 + (t1 - t0) * acc / sum(weight)
        i = j
    end = max(total, starts[-1] + 1.0)
    cuts = [0.0]
    for i in range(1, n):
        cuts.append(round(min(max(starts[i] - LEAD, cuts[-1] + 0.4), end - 0.4), 2))
    for i, s in enumerate(shots):
        s["start"] = cuts[i]
        s["end"] = cuts[i + 1] if i + 1 < n else round(end, 2)
    return quality


def _list(labels):
    """"Shot 4 has" / "Shots 4, 9 and 12 have" for a note."""
    labels = [str(x) for x in labels]
    if len(labels) == 1:
        return f"Shot {labels[0]} has"
    return "Shots " + ", ".join(labels[:-1]) + " and " + labels[-1] + " have"


def _same_list(r):
    """If these clips were placed before, use that shot list's file again (not a "Title 2")
    and keep what the clip check found, so nothing is checked twice."""
    import json
    import ss_script as S
    from ss_core import safe_name
    js = S.out_dir() / f"{safe_name(r['title'])[:80]} - shot list.json"
    try:
        old = json.loads(js.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return
    if not old.get("placed") or old.get("clip_folder") != r["clip_folder"]:
        return
    r["files"] = {"json": str(js), "text": str(js.with_name(js.name[:-5] + ".txt")),
                  "folder": str(js.parent)}
    was = {s.get("label"): s for s in old.get("shots") or []}
    for s in r["shots"]:
        if s["label"] in was and was[s["label"]].get("files"):
            s["files"] = was[s["label"]]["files"]
    for k in ("clip_checked", "continuity", "clip_notes"):
        if old.get(k):
            r[k] = old[k]


# ---------------------------------------------------------------------------
# the whole job
# ---------------------------------------------------------------------------
def build(spec, progress=None, cancel=None):
    import ss_claude as C
    import ss_roughcut as RC
    say = progress or (lambda f, m: None)
    folder = (spec.get("folder") or "").strip()
    if not folder or not Path(folder).is_dir():
        raise StoryError("Choose the folder with your clips first.")
    kind = spec.get("kind") if spec.get("kind") in ("story", "music", "cartoon") else "story"
    script = (spec.get("script") or "").strip()
    audio = (spec.get("audio") or "").strip()
    voices = (spec.get("voices") or "").strip()
    if audio and not Path(audio).exists():
        raise StoryError("That sound file isn't there any more. Choose it again.")
    if voices and not Path(voices).is_dir():
        raise StoryError("That folder of voice files isn't there any more. Choose it again.")
    say(0.02, "Looking for your clips…")
    found, unknown, _, _ = look(folder)
    if not found:
        raise StoryError("StoryStudio didn't find any clips or pictures with a shot number in "
                         "their name in that folder. Name them like “Shot 5.mp4”.")
    work = APP_DATA / "work" / "place"
    work.mkdir(parents=True, exist_ok=True)
    shots = []
    for n, d in enumerate(found, 1):
        p = Path(d["clip"] or d["picture"])
        f = {"path": str(p), "name": p.name, "kind": "clip" if d["clip"] else "picture"}
        shots.append({"n": n, "label": d["label"], "scene": 1, "title": "", "purpose": "",
                      "words": "", "picture": "", "motion": "", "camera": "", "camera2": "",
                      "characters": [], "extra": False, "start": 0.0, "end": 0.0,
                      "picture_prompt": "", "minimax_prompt": "",
                      "paths": [x for x in (d["clip"], d["picture"]) if x], "_file": f})

    def measure(s):
        f = s["_file"]
        if f["kind"] == "clip":
            try:
                info = probe(f["path"])
                f.update(seconds=info.get("duration") or 0, w=info.get("width") or 0,
                         h=info.get("height") or 0)
            except Exception:
                log_error("placing: measuring " + f["name"])
        return s
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(measure, shots))
    if cancel and cancel():
        raise StoryError("Stopped.")
    notes = []
    title = (spec.get("title") or "").strip() or Path(folder).name
    summary = ""
    if script and C.provider():
        say(0.06, "Taking a still from every clip…")

        def still(s):
            s["_still"] = _still(s["_file"], work, f"s{s['n']:03d}")
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(still, shots))
        try:
            summary = match(kind, script, shots, work,
                            lambda f, m: say(0.12 + 0.5 * f, m), cancel)
        except StoryError as e:
            if "Stopped" in str(e):
                raise
            log("placing: matching skipped:", e)
            notes.append("The AI couldn't read the script this time, so your clips are in "
                         "number order without their words. " + str(e))
    elif script:
        notes.append("Matching clips to the script needs Claude or ChatGPT (set one up in "
                     "Settings, the gear at the top). For now your clips are in number order.")
    clips = [s["_file"] for s in shots if s["_file"]["kind"] == "clip" and
             s["_file"].get("seconds")]
    lengths = sorted(f["seconds"] for f in clips)
    typical = lengths[len(lengths) // 2] if lengths else 5.0
    timed, total, joined = False, 0.0, None
    if voices and kind != "cartoon":                       # many voice files: join them first
        audio = ""
        if not any(s["words"] for s in shots):
            notes.append("StoryStudio can only put separate voice files in order once it "
                         "knows the script's words, so they were left out this time.")
        else:
            from ss_core import safe_name
            try:
                out = Path(spec.get("out_dir") or folder) / \
                    f"{safe_name(title)} - narration {V.JOINED}.wav"
                heard, total, joined = V.join(voices, shots, out,
                                              lambda f, m: say(0.64 + 0.28 * f, m), cancel)
                audio = str(out)
                quality = time_shots(shots, heard, total)
                timed = quality >= GOOD_MATCH
                notes.append(f"Your {joined['files']} voice files are joined in script order "
                             f"into one new file, {joined['name']}, in {joined['pieces']} "
                             "pieces with a short pause between them. Your own files aren't "
                             "changed."
                             + (" Not used, because the same line is in another file: "
                                + ", ".join(joined["doubles"][:8])
                                + ("…" if len(joined["doubles"]) > 8 else "") + "."
                                if joined["doubles"] else "")
                             + (" Left out because their words weren't found in the script: "
                                + ", ".join(joined["left_out"][:8])
                                + ("…" if len(joined["left_out"]) > 8 else "") + "."
                                if joined["left_out"] else ""))
                if joined["found"] < 0.6:
                    notes.append(f"Only about {int(joined['found'] * 100)}% of the script's "
                                 "words were found in those voice files, so some lines are "
                                 "probably missing from the narration.")
                if not timed:
                    notes.append("The clips are back to back in order, because too little of "
                                 "the script was found in the voices to time them.")
            except StoryError as e:
                if "Stopped" in str(e):
                    raise
                notes.append(str(e) + " The clips are back to back in order.")
            except Exception:
                log_error("placing: joining the voices")
                notes.append("StoryStudio couldn't join those voice files, so the clips are "
                             "back to back in order.")
    elif audio:
        try:
            total = probe(audio).get("duration") or 0.0
        except Exception:
            total = 0.0
    if joined:
        pass
    elif audio and total and kind != "cartoon" and any(s["words"] for s in shots):
        say(0.64, "Listening to your recording to find when each line is said…")
        try:
            heard = M.transcribe_raw(M.wav_mono16k(audio),
                                     lambda f: say(0.64 + 0.28 * f, "Listening to your "
                                                   "recording to find when each line is said…"))
            quality = time_shots(shots, heard, total)
            timed = quality >= GOOD_MATCH
            log(f"placing: {len(heard)} words heard, {quality:.0%} of the script found")
            if not timed:
                notes.append("StoryStudio couldn't follow the script in that recording "
                             + ("(singing over music is hard to hear word by word), "
                                if kind == "music" else "(is it the right recording?), ")
                             + "so the clips are back to back in order. The recording is on "
                             "the timeline for you to line them up.")
        except Exception:
            log_error("placing: listening")
            notes.append("StoryStudio couldn't listen to that recording, so the clips are "
                         "back to back in order.")
    planned = [s["_plan"] for s in shots if s.get("_plan")]
    timed_by = "recording" if timed else ""
    if not timed and len(planned) >= 0.8 * len(shots) and \
            all(b[0] >= a[0] for a, b in zip(planned, planned[1:])):
        t, shift = 0.0, 0.0                                # a shot the script doesn't have
        for s in shots:                                    # is fitted in, and moves the rest
            if s.get("_plan"):
                a_t, b_t = s["_plan"]
                s["start"] = round(max(t, a_t + shift), 2)
                s["end"] = round(max(s["start"] + 0.5, b_t + shift), 2)
            else:
                sec = s["_file"].get("seconds") or RC.HOLD
                s["start"], s["end"] = round(t, 2), round(t + sec, 2)
                shift += sec
            t = s["end"]
        timed, timed_by = True, "script"
    if not timed:
        t = 0.0
        for s in shots:
            sec = s["_file"].get("seconds") or RC.HOLD
            s["start"], s["end"] = round(t, 2), round(t + sec, 2)
            t += sec
            if s["words"] and kind == "story":             # how long the words take to say
                s["say"] = round(len(s["words"].split()) * SAY, 1)
        tight = [s["label"] for s in shots if s.get("say") and s["_file"]["kind"] == "clip"
                 and s["say"] > (s["_file"].get("seconds") or 0) / RC.SLOW]
        if tight:
            notes.append(_list(tight) + " more words than the clip can cover, even slowed "
                         "down a little. Plan another shot or a picture there when you add "
                         "the narration.")
    for s in shots:
        s.pop("_file", None)
        s.pop("_still", None)
        s.pop("_plan", None)
    a, b = next(((f["w"], f["h"]) for f in clips if f.get("w") and f.get("h")), (16, 9))
    aspect = "9:16" if b > a * 1.2 else "1:1" if abs(a - b) < a * 0.1 else "16:9"
    r = {"kind": kind, "title": title, "summary": summary, "placed": True, "timed": timed,
         "timed_by": timed_by,
         "spec": {"kind": kind, "title": title, "placed": True},
         "aspect": aspect, "clip": int(round(typical)) or 5,
         "seconds": round(max(total, shots[-1]["end"]), 1),
         "style": {"name": ""}, "characters": [], "scenes": [], "shots": shots,
         "main_shots": len(shots), "extra_shots": 0, "made": time.strftime("%Y-%m-%d %H:%M"),
         "clip_folder": str(folder), "audio": audio, "script": script, "voices": voices,
         "voice_order": (joined or {}).get("order") or [],
         "place_notes": notes, "clip_unknown": unknown[:40]}
    _same_list(r)
    say(0.95, "Putting your clips on the timeline…")
    RC.build(r, folder, spec.get("out_dir"))               # also saves the shot list
    say(1.0, "Done.")
    log("placing:", title, len(shots), "shots; timed" if timed else "shots; in order")
    return r
