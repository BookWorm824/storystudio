"""Checking the pictures and clips made for a shot list.

    check(r, folder, progress, cancel)

The user makes a picture and then a video clip for every shot, and names each file with its
shot number ("shot 5.png", "shot 5.mp4", "shot 5b.mp4" for another try). StoryStudio finds the
files, looks at every new one (a clip as a run of stills), and says for each: keep it, use
only part of it, or make it again, with a corrected prompt to paste. The answers are kept on
the shots (shot["files"]) so the shot list also shows what is done and what is left.
"""
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ss_core import APP_DATA, StoryError, log, log_error, probe
import ss_media as M
import ss_craft as craft

PICTURES = {".png", ".jpg", ".jpeg", ".webp"}
CLIPS = {".mp4", ".mov", ".webm", ".m4v", ".mkv"}
NAMED = re.compile(r"(?:shot|sh|clip|scene)[\s_\-#.]*0*(\d{1,3})(?!\d)", re.I)
LEADING = re.compile(r"^\s*0*(\d{1,3})(?!\d)")


def shot_number(name):
    """"shot 12b take 2.mp4" -> 12; "07 - the splash.png" -> 7; None if it has no number."""
    stem = Path(name).stem
    m = NAMED.search(stem) or LEADING.match(stem)
    return int(m.group(1)) if m else None


def find_files(r, folder):
    """{shot number: [paths]}, {character name: picture path}, [files with no shot number]."""
    if r.get("placed"):                 # made from finished clips: each shot has its files
        found = {}
        for s in r["shots"]:
            mine = [p for p in s.get("paths") or [] if Path(p).exists()]
            clips = [p for p in mine if Path(p).suffix.lower() in CLIPS]
            found[s["n"]] = clips or mine                # its picture only stands in
        return found, {}, list(r.get("clip_unknown") or [])
    folder = Path(folder)
    if not folder.is_dir():
        raise StoryError("That folder couldn't be opened.")
    names = {c["name"].strip().lower(): c["name"] for c in r.get("characters") or []}
    numbers = {s["n"] for s in r["shots"]}
    shots, chars, unknown = {}, {}, []
    files = [p for p in folder.iterdir() if p.is_file()]
    for sub in folder.iterdir():                       # one level of folders inside it
        if sub.is_dir() and sub.name.lower() != "storystudio":
            try:
                files += [p for p in sub.iterdir() if p.is_file()]
            except OSError:
                continue
    for p in sorted(files, key=lambda p: p.name.lower()):
        ext = p.suffix.lower()
        if ext not in PICTURES and ext not in CLIPS:
            continue
        if "(karaoke)" in p.name.lower() or "(lyrics)" in p.name.lower() or \
                "storystudio" in p.name.lower():
            continue
        stem = p.stem.strip().lower()
        who = next((names[k] for k in names if stem == k or stem.startswith(k + " ") or
                    stem.startswith(k + "_") or stem.startswith(k + "-")), None)
        if who and ext in PICTURES and shot_number(p.name) is None:
            chars.setdefault(who, str(p))
            continue
        n = shot_number(p.name)
        if n in numbers:
            shots.setdefault(n, []).append(str(p))
        else:
            unknown.append(p.name)
    return shots, chars, unknown


# ---------------------------------------------------------------------------
# what the AI is asked
# ---------------------------------------------------------------------------
CHECK = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "verdict": {"type": "string", "enum": ["keep", "use part", "make again"],
                    "description": "keep = good as it is; use part = only some seconds are "
                                   "good (clips only); make again = not usable"},
        "problems": {"type": "array", "items": {"type": "string"},
                     "description": "each problem in a few plain words, with the second it "
                                    "shows for a clip; empty when there are none"},
        "best_from": {"type": "number", "description": "clips: second where the good part "
                      "starts (0 for pictures)"},
        "best_to": {"type": "number", "description": "clips: second where the good part "
                    "ends (0 for pictures)"},
        "speed": {"type": "integer", "description": "clips: speed to play it at in the "
                  "edit, in percent (100 = as it is; 120-150 when the movement floats)"},
        "what_to_do": {"type": "string", "description": "one or two plain, kind sentences: "
                       "what to do with this file"},
        "new_prompt": {"type": "string", "description": "only for 'make again': the prompt "
                       "to paste to make it again, rewritten so the problem doesn't come "
                       "back; otherwise empty"},
        "redo": {"type": "string", "enum": ["", "picture", "clip"],
                 "description": "for 'make again': which one to make again. 'picture' when "
                                "the fault is already in the still picture (a clip can't "
                                "fix that); 'clip' when the picture is fine but the movement "
                                "went wrong; empty otherwise"},
    },
    "required": ["id", "verdict", "problems", "best_from", "best_to", "speed", "what_to_do",
                 "new_prompt", "redo"],
    "additionalProperties": False,
}
CHECKS = {"type": "object", "properties": {"files": {"type": "array", "items": CHECK}},
          "required": ["files"], "additionalProperties": False}
SYSTEM = """You check the pictures and video clips a storyteller has made with AI for her \
short film, one shot at a time, so she doesn't waste time or credits. She is not a \
professional: plain, warm, everyday words, no jargon. Be honest and specific; never invent a \
problem, and never fail a file for something a viewer wouldn't notice.

A video clip is shown as a run of stills in order, with the second printed on each. A clip is \
made by animating her still picture, so the first still is (nearly) her picture.

What to look for in a PICTURE:
- Does it show what the shot asks for: the right subject, action, place and framing?
- Does each character match their fixed look (and their reference picture when one is given): \
species, colours, markings, clothes, proportions? The same art style as the other pictures?
- Faults an image AI makes: extra or missing limbs, fingers, ears or tails; a melted or doubled \
face; nonsense letters or signs; objects fused together; a second copy of the character.
- Is the pose a good START for the movement asked for? A picture that already shows the end \
of the action leaves the video nothing to do.

What to look for in a CLIP:
- Did the action that was asked for happen, clearly, and only once (not looping or reversing)?
- Does anything morph: the face, eyes, paws, ears, tail or colours changing shape between \
stills; things appearing, vanishing or melting; a second character growing in?
- Is the movement alive, or does it float like slow motion, slide, or barely move?
- Did the camera do what was asked (a still camera staying still)?
- Is a mouth moving as if talking when nobody should speak?
- Where does it start being good and stop being good? AI clips often start slow and go wrong \
near the end. If at least a second and a half shows the action cleanly, the verdict is "use \
part" with those seconds, not "make again": trimming is free, a new clip is not.
- If the movement floats but is otherwise fine, keep it and give a speed over 100.

When something must be made again, write the new prompt yourself:
- A new CLIP prompt (image-to-video): 1-2 short sentences, the subject first and ONE clear \
action with its speed and where it ends, then the camera in plain words. Say what the problem \
part should do instead in positive words (a still camera: "the camera stays perfectly still"; \
a closed mouth: "mouth closed"); never write "no ..." or "don't ...". Keep her wording where \
it worked.
- A new PICTURE prompt (text-to-image): her picture description with the fault fixed, the \
character described by their fixed look word for word, the pose at the START of the action.
Never use a character's name in a prompt (the AI doesn't know names): say "the girl", "the \
puppy", or their look. Never add the art style; it is added for her."""


def _about(r, s):
    kind = {"story": "narrated story", "cartoon": "silent slapstick cartoon"}.get(
        r["kind"], "music video")
    cast = {c["name"].lower(): c for c in r.get("characters") or []}
    who = "; ".join(f"{n}: {cast[n.lower()].get('short') or cast[n.lower()]['look']}"
                    for n in s.get("characters") or [] if n.lower() in cast)
    return (f"SHOT {s['n']} ({kind}): {s.get('title', '')}. {s.get('purpose', '')}\n"
            f"  {'What the picture shows' if r.get('placed') else 'The picture asked for'}: "
            f"{s.get('picture', '')}\n"
            f"  The movement asked for: {s.get('motion', '')} [camera: {s.get('camera', '')}"
            f"{', ' + s['camera2'] if s.get('camera2') else ''}]\n"
            + (f"  Characters in it: {who}\n" if who else "")
            + (f"  It has {float(s['end']) - float(s['start']):.1f} seconds in the film: "
               "best_from and best_to should be the best stretch of about that length "
               f"(up to {(float(s['end']) - float(s['start'])) * 1.25:.1f} seconds, since it "
               "can play a little faster).\n" if r.get("timed") and
               float(s.get("end") or 0) > float(s.get("start") or 0) else "")
            + (f"  Sounds: {s['words']}\n" if r["kind"] == "cartoon" and s.get("words") else ""))


def _small(path, work, name, width=960):
    """A smaller jpg copy of a picture for the AI to look at."""
    from PIL import Image
    out = Path(work) / f"{name}.jpg"
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, max(1, round(im.height * width / im.width))))
    im.save(out, quality=85)
    return str(out)


def check(r, folder, progress=None, cancel=None, again=False, per_request=5, workers=3):
    """Find the pictures and clips in the folder, check every new or changed one, and keep
    the answers on the shots. Returns a summary dict."""
    import ss_claude as C
    import ss_script as S
    say = progress or (lambda f, m: None)
    say(0.02, "Looking for your pictures and clips…")
    found, chars, unknown = find_files(r, folder)
    work = APP_DATA / "work" / "clipcheck"
    work.mkdir(parents=True, exist_ok=True)
    jobs = []
    for s in r["shots"]:
        old = {f["path"]: f for f in s.get("files") or []}
        files = []
        for path in found.get(s["n"], []):
            p = Path(path)
            kind = "clip" if p.suffix.lower() in CLIPS else "picture"
            f = {"path": path, "name": p.name, "kind": kind, "mtime": int(p.stat().st_mtime)}
            was = old.get(path)
            if was and was.get("mtime") == f["mtime"] and was.get("check") and not again:
                f = was                                   # already checked, not changed
            else:
                if kind == "clip":
                    try:
                        info = probe(path)
                        f.update(seconds=round(info.get("duration") or 0, 2),
                                 fps=round(info.get("fps") or 0, 2),
                                 w=info.get("width") or 0, h=info.get("height") or 0)
                    except Exception:
                        f.update(seconds=0, fps=0, w=0, h=0)
                jobs.append((s, f))
            files.append(f)
        s["files"] = files
    r["clip_folder"] = str(folder)
    r["clip_unknown"] = unknown[:40]
    if cancel and cancel():
        raise StoryError("Stopped.")
    if jobs and not C.provider():
        raise StoryError("Checking pictures and clips needs Claude or ChatGPT. Open Settings "
                         "(the gear at the top) to set one up.")

    def prepare(k):
        s, f = jobs[k]
        name = f"f{k:03d}"
        try:
            if f["kind"] == "clip" and f.get("seconds", 0) > 0.3:
                frames = M.burst(f["path"], 0.0, f["seconds"], 12, work, name, width=480)
                f["_sheets"] = M.contact_sheets(frames, work / "close", per_sheet=12, cols=4,
                                                label=name + "_", fmt=lambda t: f"{t:.2f}s")
            elif f["kind"] == "picture":
                f["_small"] = _small(f["path"], work, name)
        except Exception:
            log_error("clip check: preparing " + f["name"])
        return k
    if jobs:
        say(0.05, f"Getting {len(jobs)} file{'s' if len(jobs) != 1 else ''} ready to look at…")
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(prepare, range(len(jobs))))
    ready = [(k, s, f) for k, (s, f) in enumerate(jobs) if f.get("_sheets") or f.get("_small")]
    groups = [ready[i:i + per_request] for i in range(0, len(ready), per_request)]
    refs = {n: _small(p, work, f"ref_{i}") for i, (n, p) in enumerate(chars.items())}
    lock, done = threading.Lock(), [0]

    def one(grp):
        if cancel and cancel():
            return
        parts, images, used = [], [], set()
        for k, s, f in grp:
            for n in s.get("characters") or []:
                if n in refs and n not in used:
                    used.add(n)
                    images.append((f"Reference picture of {n} (how {n} must look):", refs[n]))
            if f["kind"] == "clip":
                parts.append(f"FILE id {k}: a CLIP, \"{f['name']}\", {f['seconds']:.1f} seconds "
                             f"long, for:\n{_about(r, s)}")
                for a, b, p in f["_sheets"]:
                    images.append((f"File {k} (clip), stills from {a:.2f}s to {b:.2f}s:", p))
            else:
                parts.append(f"FILE id {k}: a PICTURE, \"{f['name']}\", for:\n{_about(r, s)}")
                images.append((f"File {k} (picture):", f["_small"]))
        prompt = (f"\"{r['title']}\": {r.get('summary', '')}\n"
                  f"Art style of every picture: {r['style'].get('name', '')}.\n\n"
                  + "\n".join(parts) +
                  "\nCheck every file above and answer with its id. For a picture, best_from "
                  "and best_to are 0 and speed is 100."
                  + ("\n" + craft.CLIPS if r["kind"] == "cartoon" else ""))
        try:
            res = C.ask(prompt, CHECKS, system=SYSTEM, images=images, timeout=900)
        except StoryError as e:
            log("clip check skipped for", len(grp), "files:", e)
            res = {}
        except Exception:
            log_error("clip check")
            res = {}
        with lock:
            byid = {str(a.get("id")): a for a in res.get("files", [])}
            for k, s, f in grp:
                a = byid.get(str(k))
                if not a:
                    continue
                c = {"verdict": a["verdict"], "problems": [x for x in a["problems"] if x][:5],
                     "what_to_do": a["what_to_do"].strip(), "new_prompt": "", "redo": ""}
                if f["kind"] == "clip":
                    lo = max(0.0, min(float(a.get("best_from") or 0), f["seconds"]))
                    hi = max(lo, min(float(a.get("best_to") or f["seconds"]), f["seconds"]))
                    if a["verdict"] == "keep" and hi - lo < 0.5:
                        lo, hi = 0.0, f["seconds"]
                    c.update(best_from=round(lo, 2), best_to=round(hi, 2),
                             speed=max(50, min(300, int(a.get("speed") or 100))))
                    if a["verdict"] == "use part" and hi - lo < 1.0:
                        c["verdict"] = "make again"
                else:
                    if c["verdict"] == "use part":
                        c["verdict"] = "keep"
                if c["verdict"] == "make again":
                    c["new_prompt"] = (a.get("new_prompt") or "").strip()
                    c["redo"] = a.get("redo") or ("picture" if f["kind"] == "picture"
                                                  else "clip")
                f["check"] = c
            done[0] += 1
            say(0.1 + 0.85 * done[0] / max(1, len(groups)),
                f"{C.ai_name()} is looking at your pictures and clips "
                f"({done[0]} of {len(groups)})…")
    if groups:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, groups))
    for _, f in jobs:
        f.pop("_sheets", None)
        f.pop("_small", None)
    for s in r["shots"]:
        _status(r, s)
    if jobs or again or "continuity" not in r:           # something new: look across the shots
        try:
            say(0.96, "Checking that the shots match each other…")
            r["continuity"] = continuity(r, work, lambda m: say(0.97, m), cancel)
        except Exception:
            log_error("continuity look")
    r["clip_notes"] = _technical(r)
    r["clip_checked"] = time.strftime("%Y-%m-%d %H:%M")
    r["files"] = S.save(r)
    say(1.0, "Done.")
    log("clip check:", len(jobs), "new files;", summary(r))
    return summary(r)


# ---------------------------------------------------------------------------
# continuity: every shot's picture side by side, in order
# ---------------------------------------------------------------------------
CONTINUITY = {
    "type": "object",
    "properties": {"issues": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "shot": {"type": "integer", "description": "the shot that should change"},
            "against": {"type": "integer", "description": "the shot it doesn't match"},
            "kind": {"type": "string", "enum": ["character", "place", "light", "direction",
                                                "colour", "style", "size"]},
            "problem": {"type": "string", "description": "what differs, in plain words"},
            "fix": {"type": "string", "description": "what to do, plainly: which picture to "
                    "make again and what to ask for, or how to fix it in the edit"},
        },
        "required": ["shot", "against", "kind", "problem", "fix"],
        "additionalProperties": False}}},
    "required": ["issues"], "additionalProperties": False,
}
CONTINUITY_SYSTEM = """You are the continuity checker for a short film made with AI pictures. \
You see one small picture per shot, in story order, each labelled with its shot number. Each \
picture was made separately, so things drift. Find only differences a viewer would notice \
when the shots are cut together, and say how to fix each. Plain, warm, everyday words.

Look for:
- CHARACTER: the same character looking different from one shot to another (face, hair, fur \
colour or markings, clothes, proportions, age).
- PLACE: the same place changing (furniture or doors moving, a different room, things \
appearing or vanishing) between shots of one scene.
- LIGHT: the time of day or the light jumping inside one scene (sunny to dusk and back).
- DIRECTION: a character travelling or looking toward the left in one shot and toward the \
right in the next for no reason, or two characters who should be facing each other both \
facing the same way. (Mirroring the picture in the edit often fixes this for free: say so.)
- COLOUR: one picture much brighter, darker, warmer or cooler than its neighbours (this can \
often be matched in the edit: say so).
- STYLE: a picture drawn in a different style from the rest.
- SIZE: a character or object much bigger or smaller than in the shots around it.

Different shots of a scene SHOULD differ in framing and action; that is not a problem. Report \
the picture that is the odd one out as "shot" and the one it should match as "against". Only \
real, visible problems: an empty list is a good answer."""


def continuity(r, work, say=None, cancel=None):
    """One look across all the shots that have a picture or a clip. Returns the issues."""
    import ss_claude as C
    frames, by_scene = [], {}
    for s in r["shots"]:
        files = s.get("files") or []
        f = next((x for x in files if x["path"] == s.get("best")), None) or \
            next((x for x in files if x["kind"] == "picture"), None) or \
            (files[0] if files else None)
        if not f:
            continue
        try:
            if f["kind"] == "clip":
                got = M.burst(f["path"], 0.0, max(0.2, min(1.0, f.get("seconds") or 1.0)), 1,
                              work, f"c{s['n']:03d}", width=480)
                path = got[0][1] if got else None
            else:
                path = _small(f["path"], work, f"c{s['n']:03d}", width=480)
        except Exception:
            log_error("continuity picture")
            path = None
        if path:
            frames.append((float(s["n"]), path))
            by_scene.setdefault(s.get("scene", 1), []).append(s["n"])
    if len(frames) < 2:
        return []
    from PIL import Image
    w0, h0 = Image.open(frames[0][1]).size             # the sheet needs equal sizes
    for _, path in frames:
        im = Image.open(path)
        if im.size != (w0, h0):
            im.convert("RGB").resize((w0, h0)).save(path, quality=85)
    sheets = M.contact_sheets(frames, Path(work) / "close", per_sheet=12, cols=4,
                              label="cont_", fmt=lambda n: f"Shot {int(n)}")
    scenes = "; ".join(f"scene {k}: shots {', '.join(map(str, v))}"
                       for k, v in sorted(by_scene.items()))
    cast = "\n".join(f"- {c['name']}: {c.get('short') or c['look']}"
                     for c in r.get("characters") or []) or "(no recurring characters)"
    numbers = {int(n) for n, _ in frames}
    issues = []
    for i in range(0, len(sheets), 4):                  # up to 48 shots a request
        if cancel and cancel():
            break
        part = sheets[i:i + 4]
        if say:
            say(f"{C.ai_name()} is checking that the shots match each other…")
        try:
            res = C.ask(f"\"{r['title']}\": {r.get('summary', '')}\n"
                        f"Which shots belong to which scene: {scenes}\n"
                        f"The characters' fixed looks:\n{cast}\n\n"
                        "Check these shots against each other.", CONTINUITY,
                        system=CONTINUITY_SYSTEM,
                        images=[(f"Shots {int(a)} to {int(b)}:", p) for a, b, p in part],
                        timeout=600)
        except StoryError as e:
            log("continuity look skipped:", e)
            continue
        for x in res.get("issues", []):
            if x.get("shot") in numbers and x.get("against") in numbers and \
                    x["shot"] != x["against"]:
                issues.append({k: x[k] for k in ("shot", "against", "kind", "problem", "fix")})
    return issues[:20]


RANK = {"keep": 0, "use part": 1, "make again": 2}


def _status(r, s):
    """The shot's status from its files: 'todo', 'picture' (a good picture, no clip yet),
    'redo picture', 'redo clip', 'done'. The best clip is kept as s['best']."""
    files = s.get("files") or []
    clips = [f for f in files if f["kind"] == "clip" and f.get("check")]
    pics = [f for f in files if f["kind"] == "picture"]
    s.pop("best", None)
    if clips:
        best = min(clips, key=lambda f: (RANK.get(f["check"]["verdict"], 3), -f["mtime"]))
        if best["check"]["verdict"] in ("keep", "use part"):
            s["best"] = best["path"]
            s["status"] = "done"
        else:
            s["status"] = "redo picture" if best["check"].get("redo") == "picture" \
                else "redo clip"
            s["best"] = best["path"]
    elif any(f["kind"] == "clip" for f in files):
        s["status"] = "unchecked"
    elif pics:
        seen = [f for f in pics if f.get("check")]
        ok = any(f["check"]["verdict"] == "keep" for f in seen)
        s["status"] = "picture" if ok or not seen else "redo picture"
        if seen and not ok:
            s["best"] = max(seen, key=lambda f: f["mtime"])["path"]
    else:
        s["status"] = "todo"


def _technical(r):
    """Plain notes on things that trip up the edit later: mixed frame rates and shapes."""
    clips = [f for s in r["shots"] for f in s.get("files") or [] if f["kind"] == "clip"
             and f.get("fps")]
    notes = []
    if not clips:
        return notes
    rates = {}
    for f in clips:
        rates.setdefault(round(f["fps"]), []).append(f["name"])
    if len(rates) > 1:
        main = max(rates, key=lambda k: len(rates[k]))
        odd = [n for k, v in rates.items() if k != main for n in v]
        notes.append(f"Most of your clips are {main} frames a second, but {len(odd)} "
                     f"{'is' if len(odd) == 1 else 'are'} different ({', '.join(odd[:4])}"
                     f"{'…' if len(odd) > 4 else ''}). Premiere copes, but those shots can "
                     "look slightly jerky. Make them at the same setting if you can.")
    try:
        a, b = (float(x) for x in str(r.get("aspect") or "16:9").split(":"))
        want = a / b
        odd = [f["name"] for f in clips if f.get("w") and f.get("h") and
               abs(f["w"] / f["h"] - want) > 0.03]
        if odd:
            notes.append(f"{len(odd)} clip{'s are' if len(odd) != 1 else ' is'} a different "
                         f"shape from the {r['aspect']} you chose ({', '.join(odd[:4])}"
                         f"{'…' if len(odd) > 4 else ''}), so there will be black bars or "
                         "cropping.")
    except (ValueError, ZeroDivisionError):
        pass
    small = [f["name"] for f in clips if f.get("h") and min(f["w"], f["h"]) < 700]
    if small:
        notes.append(f"{len(small)} clip{'s are' if len(small) != 1 else ' is'} lower than "
                     "720p and may look soft next to the others.")
    return notes


def summary(r):
    n = {"todo": 0, "picture": 0, "redo picture": 0, "redo clip": 0, "done": 0, "unchecked": 0}
    for s in r["shots"]:
        n[s.get("status", "todo")] = n.get(s.get("status", "todo"), 0) + 1
    return dict(n, shots=len(r["shots"]), folder=r.get("clip_folder", ""),
                unknown=len(r.get("clip_unknown") or []), notes=r.get("clip_notes") or [])
