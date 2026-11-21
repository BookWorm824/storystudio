"""Change a sound she doesn't like.

    sounds_at(st, t)                          the StoryStudio sounds playing at a moment
    change(st, item_id, action, t, note)      "remove", "stop" (stop it here) or "remake"

Works on her project as she last saved it, and writes the result to a copy ending in
"(sounds changed)"; a second change builds on that copy. Every sound she turns down is
remembered for this movie, so later sound runs don't plan anything like it.
A remade sound is kept in its own small run (work/changes/<n>/state.json) with a copy of the
timeline it was placed on, so "Line up my sounds again" moves it with its shot later.
"""
import json
import os
import re
import shutil
import time
from pathlib import Path

from ss_core import APP_DATA, TICKS, StoryError, fmt_time, log, log_error

KINDS = [("StoryStudio music", "music"), ("StoryStudio ambience", "background"),
         ("StoryStudio effects", "effect")]
KIND_NAMES = {"music": "Music", "background": "Background sound", "effect": "Sound effect",
              "character": "Character sound"}
DISLIKES = APP_DATA / "sound_dislikes.json"


def parse_time(text, fps=24.0, duration=None):
    """"1:47", "00:01:47:23" (hours:minutes:seconds:frames, as Premiere shows it), "107.9"."""
    t = str(text or "").strip()
    if not t:
        raise StoryError("Type the time first, like 1:47 or 00:01:47:23.")
    try:
        nums = [float(p) for p in re.split(r"[:;]", t)]
    except ValueError:
        raise StoryError("StoryStudio couldn't read that time. Type it the way Premiere shows "
                         "it (00:01:47:23) or like 1:47.")
    if len(nums) == 4:
        h, m, s, f = nums
        return h * 3600 + m * 60 + s + f / (fps or 24.0)
    if len(nums) == 3:
        h, m, s = nums
        if duration and h * 3600 + m * 60 + s > duration + 5:   # minutes:seconds:frames
            return h * 60 + m + s / (fps or 24.0)
        return h * 3600 + m * 60 + s
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    return nums[0]


def kind_of(tname, name=""):
    for prefix, k in KINDS:
        if tname.startswith(prefix):
            if k == "effect" and not name.startswith("Shot "):
                return "character"
            return k
    return None


# --- the changes she has made so far ---------------------------------------------
# Every change is written down (work/sound_changes.json). The next change starts from the
# project as she last saved it and makes the earlier ones again first, so nothing is lost when
# she saves in Premiere between two changes, or StoryStudio restarts. A change that is already
# in the project (she carried on in the changed copy) is skipped.
OPS_FILE = "sound_changes.json"
OPS_HOURS = 12


def _ops(st):
    try:
        ops = json.loads((Path(st["work"]) / OPS_FILE).read_text(encoding="utf8"))
    except (OSError, ValueError):
        return []
    me = os.path.normcase(str(st["project"]))
    return [o for o in ops if os.path.normcase(o.get("project", "")) == me and
            time.time() - o.get("when", 0) < OPS_HOURS * 3600]


def _save_ops(st, ops):
    try:
        (Path(st["work"]) / OPS_FILE).write_text(json.dumps(ops, indent=1), encoding="utf8")
    except OSError:
        log_error("saving sound changes")


def _find(items, name, start, slack=1.5):
    """StoryStudio's clip with this name nearest a time (seconds), or None."""
    cs = [i for i in items if i["kind"] == "audio" and kind_of(i["tname"], i["name"]) and
          i["name"] == name]
    if not cs:
        return None
    c = min(cs, key=lambda i: abs(i["start"] / TICKS - start))
    return c if abs(c["start"] / TICKS - start) <= slack else None


def _drop(w, seq, item):
    import ss_prwrite as W
    lst = item["track"].find("ClipTrack/ClipItems/TrackItems")
    for r in list(lst):
        if (r.get("ObjectRef") or r.get("ObjectURef")) == item["id"]:
            lst.remove(r)
    for k, r in enumerate(lst):
        r.set("Index", str(k))
    W._unlink(w, seq, [item["id"]])


def _stop_at(w, seq, item, at):
    import ss_prwrite as W
    if at - item["start"] < TICKS * 0.3:
        _drop(w, seq, item)
        return
    W._set_times(item["cti"], item["start"], at)
    if item["c"].find("OutPoint") is not None:
        W.Writer.setx(item["c"], "OutPoint", item["in"] + int((at - item["start"]) *
                                                              item["speed"]))


def _insert(w, seq, items, new, project_dir):
    """Put a remade sound on a StoryStudio track of its kind where nothing else is playing."""
    prefix = new["track"]
    a, b = int(round(new["at"] * TICKS)), int(round((new["at"] + new["dur"]) * TICKS))
    lanes = {}
    for i in items:
        if i["kind"] == "audio" and i["tname"].startswith(prefix):
            lanes.setdefault(id(i["track"]), (i["track"], []))[1].append((i["start"], i["end"]))
    free = next((tr for tr, spans in lanes.values()
                 if all(e2 <= a or s2 >= b for s2, e2 in spans)), None)
    if free is None:
        free = w.add_track(seq, f"{prefix} {len(lanes) + 1}")
    tmpl, _ = w.template_item(".wav")
    if tmpl is None:
        tmpl = w.spare_template()
    if tmpl is None:
        raise StoryError("This project has no sound clip StoryStudio can copy, so the new "
                         "sound couldn't go in. It is in the StoryStudio sounds folder.")
    w.add_clip(free, tmpl, new["file"], new["at"], new["dur"], new["name"],
               bin_el=w.make_bin("StoryStudio sounds"), project_dir=project_dir)


def _apply(w, seq, items, op, project_dir):
    """Make one change (again). Returns True when it changed something."""
    it = _find(items, op["name"], op["start"])
    if op["action"] == "remove":
        if it is None:
            return False
        _drop(w, seq, it)
        return True
    if op["action"] == "stop":
        cut = int(round(op["t"] * TICKS))
        if it is None or not (it["start"] < cut < it["end"] - TICKS // 50):
            return False
        _stop_at(w, seq, it, cut)
        return True
    if op["action"] == "remake":
        new = op["new"]
        if any(i["kind"] == "audio" and i["name"] == new["name"] and
               abs(i["start"] / TICKS - new["at"]) < 1.5 for i in items):
            return False                           # already in
        if not os.path.exists(new["file"]):
            return False
        if it is not None:                         # the old one stops where the new one starts
            if new["at"] * TICKS > it["start"] + TICKS:
                _stop_at(w, seq, it, int(round(new["at"] * TICKS)))
            else:
                _drop(w, seq, it)
        _insert(w, seq, [i for i in items if i is not it], new, project_dir)
        return True
    return False


def opened(st):
    """The project as she last saved it, with her earlier changes made again:
    (writer, sequence, every clip)."""
    import ss_prwrite as W
    w = W.Writer(st["project"])
    seq = w.sequence(st["seq_uid"])
    items = W._all_items(w, seq)
    for op in _ops(st):
        try:
            if _apply(w, seq, items, op, str(Path(st["project"]).parent)):
                items = W._all_items(w, seq)
        except StoryError:
            raise
        except Exception:
            log_error("making an earlier sound change again")
    return w, seq, items


def sounds_at(st, t):
    w, seq, items = opened(st)
    out = []
    for i in items:
        if i["kind"] != "audio":
            continue
        k = kind_of(i["tname"], i["name"])
        if not k:
            continue
        a, b = i["start"] / TICKS, i["end"] / TICKS
        if a - 0.5 <= t <= b + 0.25:
            out.append({"id": i["id"], "raw": i["name"],
                        "name": re.sub(r"^Music:\s*", "", i["name"]), "kind": k,
                        "kind_name": KIND_NAMES[k], "start": round(a, 2), "end": round(b, 2),
                        "time": fmt_time(a), "until": fmt_time(b),
                        "long": k in ("music", "background")})
    order = {"music": 0, "background": 1, "effect": 2, "character": 3}
    out.sort(key=lambda x: (order[x["kind"]], x["start"]))
    return out


# --- remembering what she turned down -----------------------------------------
def _dislikes():
    try:
        return json.loads(DISLIKES.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return {}


def remember(seq_uid, name, kind, note):
    d = _dislikes()
    lst = d.setdefault(seq_uid, [])
    lst.append({"what": name, "kind": kind, "note": note or "", "when": time.strftime("%Y-%m-%d")})
    d[seq_uid] = lst[-500:]
    DISLIKES.write_text(json.dumps(d, indent=1), encoding="utf8")


def dislikes_text(seq_uid):
    """For the sound planner: what she didn't like in this movie before."""
    lst = _dislikes().get(seq_uid) or []
    if not lst:
        return ""
    rows = [f"- {x['what']}" + (f" (she said: {x['note']})" if x.get("note") else "")
            for x in lst[-25:]]
    return ("SOUNDS THE STORYTELLER TURNED DOWN IN THIS MOVIE (don't plan anything like these):\n"
            + "\n".join(rows))


# --- the change ------------------------------------------------------------------
def _find_cue(st, name):
    """The planned sound behind a clip on the timeline (its prompt), from any sound run."""
    import ss_audit as A
    for data, _, _ in reversed(A.sound_sources(st)):
        for c in data.get("cues") or []:
            if c.get("what") == name or f"Music: {c.get('what')}" == name:
                return c, data
    return None, None


REMAKE = {
    "type": "object",
    "properties": {
        "what": {"type": "string", "description": "a short plain name for the new sound, "
                 "3-10 words"},
        "sound_prompt": {"type": "string", "description": "the prompt for the sound AI, "
                         "10-40 words: what is heard, how, and the feeling; for music: style, "
                         "instruments, tempo and mood; never words or singing"},
        "suno_prompt": {"type": "string", "description": "for music only: the same as a "
                        "Suno style prompt; empty otherwise"},
        "search_words": {"type": "string", "description": "2-4 words to find it in a sound "
                         "library"},
        "level": {"type": "string", "enum": ["soft", "medium", "loud", "background"]},
        "reply": {"type": "string", "description": "one or two warm, plain sentences to her: "
                  "what the new sound will be and why it fits better"},
    },
    "required": ["what", "sound_prompt", "suno_prompt", "search_words", "level", "reply"],
    "additionalProperties": False,
}


def _picture(st, base, work, progress):
    """A quick picture of the timeline she's changing (kept until the project changes)."""
    import storysound as E
    import ss_media as M
    pic = work / "change picture.mp4"
    stamp = work / "change picture.txt"
    key = f"{base}|{Path(base).stat().st_mtime}"
    if pic.exists() and stamp.exists() and stamp.read_text(encoding="utf8") == key:
        return str(pic)
    seq = E._sequence({"project": st["project"], "snapshot": base, "seq_uid": st["seq_uid"]})
    made = M.build_picture(seq, work / "change_picture", lambda f: progress(
        0.05 + 0.25 * f, "Getting the picture ready…"))
    shutil.copy2(made, pic)
    stamp.write_text(key, encoding="utf8")
    return str(pic)


def change(st, item_id, action, t=None, note="", settings=None, progress=None, cancel=None,
           name="", start=None):
    """Remove a sound, stop it at t, or have a better one made. name/start (as the list showed
    them) make sure it is the clip she picked, whatever Premiere has renumbered since."""
    import storysound as E
    say = progress or (lambda f, m: None)
    w, seq, items = opened(st)
    it = next((i for i in items if i["id"] == str(item_id) and i["kind"] == "audio"
               and (not name or i["name"] == name)), None)
    if it is None and name and start is not None:
        it = _find(items, name, float(start))
    if it is None or not kind_of(it["tname"], it["name"]):
        raise StoryError("That sound has changed on the timeline since the list was made. "
                         "Press “Show the sounds there” again.")
    kind = kind_of(it["tname"], it["name"])
    a, b = it["start"], it["end"]
    frame = int(TICKS / max(1.0, float(st.get("fps") or 24)))
    cut = int(round((t or 0) * TICKS / frame)) * frame if t is not None else None
    reply = ""
    op = {"action": action, "name": it["name"], "start": a / TICKS, "note": note,
          "t": (cut or 0) / TICKS, "when": time.time(), "project": str(st["project"])}
    project_dir = str(Path(st["project"]).parent)
    if action == "remove":
        msg = f"Took out “{it['name']}”."
        gone = it["name"]
    elif action == "stop":
        if cut is None or not (a < cut < b):
            raise StoryError("Choose a moment while that sound is playing.")
        msg = f"“{it['name']}” now stops at {fmt_time(cut / TICKS)}."
        gone = f"{it['name']} (after {fmt_time(cut / TICKS)})"
    elif action != "remake":
        raise StoryError("StoryStudio doesn't know that change.")
    if action == "remake":
        import ss_claude as C
        if not C.provider():
            raise StoryError("Making a better sound needs Claude or ChatGPT. Open Settings (the "
                             "gear at the top) to set one up.")
        long = kind in ("music", "background")
        start = cut if long and cut is not None and a + TICKS < cut < b - 2 * TICKS else a
        cue0, data0 = _find_cue(st, it["name"])
        n = 1
        changes = Path(st["work"]) / "changes"
        while (changes / str(n)).exists():
            n += 1
        work = changes / str(n)
        work.mkdir(parents=True)
        snap = work / "timeline.prproj"
        shutil.copy2(st["project"], snap)
        pic = _picture(st, st["project"], Path(st["work"]), say)
        if cancel and cancel():
            raise StoryError("Stopped.")
        say(0.35, "Working out a sound that fits better…")
        import ss_media as M
        s0, s1 = start / TICKS, b / TICKS
        lo, hi = max(0.0, s0 - 1.0), min(s1 + 1.0, s0 + 12.0)
        frames = M.burst(pic, lo, hi, 12, work, "look")
        sheets = M.contact_sheets(frames, work, per_sheet=12, cols=4, label="look_",
                                  fmt=lambda x: f"{x:.1f}s")
        images = [(f"The picture from {lo:.1f}s to {hi:.1f}s:", p) for _, _, p in sheets]
        story = (data0 or {}).get("plan_notes") or ""
        others = [i for i in items if i["kind"] == "audio" and kind_of(i["tname"], i["name"]) and
                  i["id"] != it["id"] and i["end"] > start - 3 * TICKS and i["start"] < b + 3 * TICKS]
        prompt = f"""A {'silent cartoon' if (data0 or {}).get('cartoon') else 'story video'} \
called "{st['seq_name']}".
{('About the sounds: ' + story[:1500]) if story else ''}
The storyteller doesn't like this {KIND_NAMES[kind].lower()} and wants a better one:
NOW: "{it['name']}" from {fmt_time(a / TICKS)} to {fmt_time(b / TICKS)}
{('ITS PROMPT WAS: ' + cue0.get('sound_prompt', '')) if cue0 else ''}
THE NEW ONE PLAYS from {fmt_time(s0)} to {fmt_time(s1)} ({s1 - s0:.1f} s).
WHAT SHE SAID: "{note or '(nothing: she just doesn\'t like it; make it fit the story and the picture better)'}"
OTHER SOUNDS AROUND THEN: {', '.join(i['name'] for i in others[:12]) or 'none'}
{dislikes_text(st['seq_uid'])}

Look at the picture and write the new sound so it fits the story at that moment and what \
she asked for. Characters never speak: their sounds are wordless. Music must never sound like holiday music: no glockenspiel, celesta, bells, chimes, sleigh bells, jingles, choir or church organ (with warm strings they sound like a Christmas carol). Use instruments like pizzicato strings, clarinet, flute, bassoon, tuba, xylophone, marimba, ukulele, piano, woodblock and light percussion."""
        res = C.ask(prompt, REMAKE, images=images, timeout=600)
        ck = {"music": "music", "background": "ambience", "effect": "action",
              "character": "voice"}[kind]
        if ck == "music":
            E.no_holiday(res)
        cue = {"id": "c001", "kind": ck, "start": round(s0, 3), "end": round(s1, 3),
               "place_at": round(s0, 3), "length": round(s1 - s0, 3),
               "what": res["what"].strip()[:120], "sound_prompt": res["sound_prompt"].strip(),
               "adobe_prompt": res["sound_prompt"].strip(),
               "suno_prompt": (res.get("suno_prompt") or "").strip() if ck == "music" else "",
               "search_words": res.get("search_words", ""), "synced": ck not in E.BEDS,
               "level": "background" if ck in E.BEDS else res.get("level", "medium"),
               "includes": [], "keep": True, "source": "", "file": "", "note": "",
               "replaces": it["name"], "her_note": note}
        mini = {"project": st["project"], "seq_uid": st["seq_uid"], "seq_name": st["seq_name"],
                "work": str(work), "snapshot": str(snap), "picture": pic,
                "duration": st.get("duration") or (max(i["end"] for i in items) / TICKS),
                "fps": st.get("fps"), "cues": [cue], "stage": "change", "clips": [],
                "speech": [], "cartoon": bool((data0 or {}).get("cartoon"))}
        say(0.45, "Making the new sound…")
        E.make_sounds(mini, settings or {}, lambda f, m: say(0.45 + 0.4 * f, m), cancel,
                      ids=["c001"])
        if not cue.get("file") or not Path(cue["file"]).exists():
            why = cue.get("note") or "The sound couldn't be made this time."
            raise StoryError(f"{why} Nothing was changed.")
        f = cue["file"]
        _, ext = w.template_item(".wav")
        if ext and ext != ".wav":
            f2 = str(Path(f).with_suffix(ext))
            E.run([E.ffmpeg(), "-y", "-v", "error", "-i", f, "-b:a", "320k", f2])
            f = f2
        place = float(cue.get("place_at", s0))       # where the sound maker lined it up
        op["new"] = {"name": cue["what"], "file": f, "at": place,
                     "dur": E.probe(f)["duration"] or (s1 - s0),
                     "track": next(p for p, k in KINDS if it["tname"].startswith(p))}
        E.save(mini)                                    # so the new sound follows its shot later
        reply = res.get("reply", "")
        msg = f"Made “{cue['what']}” in place of “{it['name']}”."
        gone = it["name"]
        log("change sound: remade", it["name"], "->", cue["what"], "file", f)
    if not _apply(w, seq, items, op, project_dir):
        raise StoryError("That change couldn't be made: the sound isn't where the list showed "
                         "it any more. Press “Show the sounds there” again.")
    out = E.copy_path(Path(st["project"]), "sounds changed")
    w.save(out)
    _save_ops(st, _ops(st) + [op])
    remember(st["seq_uid"], gone, kind, note)
    log("change sound:", action, it["name"], "->", out)
    return {"project": str(out), "message": msg, "reply": reply}


# --- every sound's prompts, in one place ---------------------------------------------
KIND_LABEL = {"shot": ("effect", "Sound effects"), "voice": ("character", "Character sound"),
              "ambience": ("background", "Background sound"), "music": ("music", "Music")}


def prompt_list(st):
    """Every StoryStudio sound made for this movie (all runs) with its prompts, and where it is
    on the timeline now: [{...}] in timeline order, the ones not on the timeline last."""
    import ss_audit as A
    import ss_prwrite as W
    at = {}
    try:
        w = W.Writer(st["project"])
        for i in W._all_items(w, w.sequence(st["seq_uid"])):
            if i["kind"] == "audio":
                at.setdefault(i["name"], []).append(i["start"] / TICKS)
    except Exception:
        log_error("prompt list: reading the timeline")
    rows = []
    for data, proj, _ in A.sound_sources(st):
        run = ("Changed sound" if data.get("stage") == "change" else
               "New shots" if data.get("only_shots") else "Sound effects run")
        for c in data.get("cues") or []:
            if not c.get("keep", True):
                continue
            k, label = KIND_LABEL.get(c["kind"], ("effect", "Sound effect"))
            times = at.get(c["what"]) or []
            t = times.pop(0) if times else None
            sp, ap = (c.get("sound_prompt") or "").strip(), (c.get("adobe_prompt") or "").strip()
            rows.append({"what": re.sub(r"^Music:\s*", "", c["what"]), "kind": k, "label": label,
                         "run": run, "t": t, "time": fmt_time(t) if t is not None else "",
                         "made": bool(c.get("file") and Path(c["file"]).exists()),
                         "file": c.get("file") or "",
                         "sound": sp, "adobe": ap if ap and ap != sp else "",
                         "suno": (c.get("suno_prompt") or "").strip(),
                         "includes": [x for x in (c.get("includes") or []) if isinstance(x, str)],
                         "replaces": c.get("replaces", ""), "her_note": c.get("her_note", "")})
    rows.sort(key=lambda r: (r["t"] is None, r["t"] or 0))
    return rows


def write_prompt_list(st, rows=None):
    import storysound as E
    rows = rows if rows is not None else prompt_list(st)
    od = E.out_dir(st)
    od.mkdir(parents=True, exist_ok=True)
    path = od / f"{E.safe_name(st['seq_name'])} - all sound prompts.txt"
    lines = [f"StoryStudio – every sound prompt for \"{st['seq_name']}\"", "=" * 60, "",
             "Times are where each sound is on your timeline now.", ""]
    for kind, title in (("effect", "SOUND EFFECTS"), ("character", "CHARACTER SOUNDS"),
                        ("background", "BACKGROUND SOUNDS"), ("music", "MUSIC")):
        mine = [r for r in rows if r["kind"] == kind]
        if not mine:
            continue
        lines += [title, "-" * len(title)]
        for r in mine:
            lines.append(f"{r['time'] or '(not on the timeline now)'}  {r['what']}")
            if r["includes"]:
                lines.append("   In this shot: " + "; ".join(r["includes"]))
            lines.append("   Sound AI (Fal / ElevenLabs): " + r["sound"])
            if r["adobe"]:
                lines.append("   Adobe / sound library: " + r["adobe"])
            if r["suno"]:
                lines.append("   Suno: " + r["suno"])
            if r["replaces"]:
                lines.append(f"   Made in place of: {r['replaces']}" +
                             (f" (you said: {r['her_note']})" if r["her_note"] else ""))
            lines.append("")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf8")
    return str(path)
