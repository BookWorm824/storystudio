"""StoryStudio engine: Premiere project in, sound effects (synced, levelled) out.

    st = open_project(path)            # read the timeline, find the story
    analyze(st, settings, progress)    # preview, levels, speech, stills, Claude's plan
    make_sounds(st, settings, progress)  # make / find every planned sound
    finish(st, settings, progress)     # mix, preview movie, project copy, sound list

`st` is a JSON-able dict saved in the work folder after each step, so a run can be reopened.
"""
import hashlib
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np

from ss_core import (APP_DATA, StoryError, find_story, fmt_time, log, log_error, probe,
                     read_story, safe_name, run, ffmpeg)
import ss_premiere as P
import ss_media as M
import ss_sounds as S

WORK = APP_DATA / "work"
BEDS = {"ambience", "music"}        # long background layers (looped, ducked, snapped to cuts)


def gpu_free_gb():
    """Free graphics memory right now (0 if unknown). Premiere and other apps share it."""
    import subprocess
    from ss_core import NOWIN
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=5, **NOWIN)
        return float(r.stdout.strip().splitlines()[0]) / 1024
    except Exception:
        return 0.0


def estimate(st, settings, n_cues=None, gpu_gb=None):
    """Rough seconds for each step: {"prepare", "plan", "sounds", "finish"} plus a flag for
    a busy graphics card. Shown before starting and used for the time-left display."""
    d = st["duration"]
    shots = st.get("info", {}).get("shots", 30)
    prepare = (0 if st.get("export") else 8 + shots * 0.5) + 15 + d * 0.15
    sheets = math.ceil(d / (1.0 if d <= 420 else 2.0) / 20)
    plan = 90 + math.ceil(sheets / 10) * 60
    per_min = {"light": 7, "normal": 13, "lots": 24}.get(settings.get("busy", "normal"), 13)
    n = n_cues if n_cues is not None else max(4, round(d / 60 * per_min))
    beds = max(1, round(n * 0.3))
    takes = max(1, int(settings.get("best_of", 3)))
    gpu = gpu_free_gb() if gpu_gb is None else gpu_gb
    slow = 1.0 if gpu >= 7.0 else (2.0 if gpu >= 3.8 else 6.0)
    sounds = 0.0
    finish = 15 + d * 0.2
    if uses_fal(settings):                   # made by Fal, one sound per shot
        import ss_fal
        shots = [s for s in st.get("shots") or [] if s.get("name") != "black"]
        if shots:
            seconds = sum(s["end"] - s["start"] for s in shots) * 0.75   # stills are skipped
            n_shots = max(1, round(len(shots) * 0.75))
        else:
            seconds, n_shots = d * 0.7, max(4, round(d / 6))
        versions = 2 if int(settings.get("best_of", 3)) > 1 else 1
        return {"prepare": prepare, "plan": plan, "sounds": 60 + n_shots * 9.0,
                "finish": finish, "busy_gpu": False, "n": n_shots + beds, "shots": n_shots,
                "cost": ss_fal.estimate(seconds, beds * min(ss_fal.TEXT_MAX, 15.0),
                                        samples=versions, shots=True)}
    if settings.get("use_local", True):
        spots = n - (beds if settings.get("hf_key") else 0)
        sounds = 45 + spots * 8.0 * takes * slow
        if settings.get("hf_key"):
            sounds += 60 + beds * 12.0 * takes * slow
        if takes > 1:
            sounds += 25 + n * 0.6
    else:
        sounds = n * 3.0
    return {"prepare": prepare, "plan": plan, "sounds": sounds, "finish": finish,
            "busy_gpu": slow > 1.0, "n": n}


def nice_minutes(sec):
    if sec < 60:
        return "less than a minute"
    m = round(sec / 60)
    return "about 1 minute" if m <= 1 else f"about {m} minutes"


def _say(progress, frac, msg):
    if progress:
        progress(max(0.0, min(1.0, frac)), msg)


def work_dir(project_path, seq_name):
    h = hashlib.md5(f"{Path(project_path).resolve()}|{seq_name}".encode()).hexdigest()[:8]
    d = WORK / f"{safe_name(Path(project_path).stem)[:40]}-{h}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# an earlier run's sounds or edit notes are set aside, not lost, when a different kind of
# job (karaoke, fix the audio, a stopped run, a freshly opened project) saves over them
KEPT_ASIDE = {"cues": "state_sounds.json", "feedback": "state_feedback.json"}


def save(st):
    f = Path(st["work"], "state.json")
    try:
        if f.exists():
            old = json.loads(f.read_text(encoding="utf-8-sig"))
            for key, name in KEPT_ASIDE.items():
                if old.get(key) and not st.get(key):
                    shutil.copyfile(f, f.with_name(name))
    except (OSError, ValueError):
        pass
    f.write_text(json.dumps(st, indent=1), encoding="utf8")


def load(work, need=None):
    """The saved run in this work folder. need="cues" or "feedback": the latest run that
    has them (it may have been set aside by a later job of another kind)."""
    st = json.loads(Path(work, "state.json").read_text(encoding="utf-8-sig"))
    if need and not st.get(need):
        alt = Path(work, KEPT_ASIDE[need])
        if alt.exists():
            st = json.loads(alt.read_text(encoding="utf-8-sig"))
    return st


def out_dir(st):
    d = Path(st["project"]).parent / "StoryStudio"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 1. open the project
# ---------------------------------------------------------------------------
def snapshot(path, work):
    """Save StoryStudio's own copy of the project the moment it is picked, so the storyteller
    can keep working (and saving) in Premiere without changing this run. Premiere may be in
    the middle of saving, so a copy that doesn't open is retried."""
    snaps = Path(work) / "snapshots"
    snaps.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dst = snaps / f"{safe_name(Path(path).stem)[:50]} {stamp}.prproj"
    for attempt in range(4):
        shutil.copy2(path, dst)
        try:
            P.Project(dst, home=path)
            break
        except StoryError:
            if attempt == 3:
                raise StoryError("StoryStudio couldn't make its copy of that project (Premiere "
                                 "may be saving it right now). Wait a moment and pick it again.")
            time.sleep(2)
    old = sorted(snaps.glob("*.prproj"))
    for f in old[:-6]:                        # keep the last few
        try:
            f.unlink()
        except OSError:
            pass
    return str(dst)


def open_project(path, seq_uid=None, snap=None):
    path = str(Path(path).resolve())
    pre_work = work_dir(path, "")             # temporary home for the copy until we know
    if not (snap and Path(snap).exists()):    # the sequence name
        snap = snapshot(path, pre_work)
    pr = P.Project(snap, home=path)
    seqs = pr.sequences()
    if not seqs:
        raise StoryError("That project has no sequence with clips in it.")
    seq = next((s for s in seqs if s["uid"] == seq_uid), None) or pr.main_sequence()
    info = P.summary(seq)
    st = {"project": str(Path(path).resolve()), "seq_uid": seq["uid"], "seq_name": seq["name"],
          "sequences": [{"uid": s["uid"], "name": s["name"], "duration": s["duration"]}
                        for s in seqs],
          "duration": seq["duration"], "fps": seq["fps"], "width": seq["width"],
          "height": seq["height"], "info": info, "story": find_story(path, seq["name"]),
          "export": P.find_export(path, seq), "cues": [], "stage": "opened",
          "snapshot": snap, "snapshot_time": time.strftime("%I:%M %p").lstrip("0"),
          "snapshot_date": time.strftime("%Y-%m-%d")}
    st["work"] = str(work_dir(path, seq["name"]))
    if Path(snap).parent.parent != Path(st["work"]):     # keep the copy with this run
        dst = Path(st["work"]) / "snapshots" / Path(snap).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(snap, dst)
        st["snapshot"] = str(dst)
    if Path(st["work"], "state.json").exists():   # same project again: keep the old cues around
        try:
            prev = load(st["work"], "cues")
            st["previous"] = {"cues": len(prev.get("cues", [])), "stage": prev.get("stage"),
                              "feedback": bool(load(st["work"], "feedback").get("feedback"))}
        except (OSError, ValueError):
            pass
    return st


def _sequence(st):
    pr = P.Project(st.get("snapshot") or st["project"], home=st["project"])
    for el in pr.root:
        if el.tag == "Sequence" and (el.get("ObjectUID") or el.get("ObjectID")) == st["seq_uid"]:
            return pr.read_sequence(el)
    return pr.main_sequence()


# ---------------------------------------------------------------------------
# 2. analyse
# ---------------------------------------------------------------------------
def analyze(st, settings, progress=None, cancel=None):
    work = Path(st["work"])
    seq = _sequence(st)
    dur = seq["duration"]

    # picture: the exported movie if it matches, else a quick preview from the shots
    if st.get("export") and Path(st["export"]).exists():
        st["picture"] = st["export"]
        _say(progress, 0.05, "Using your exported movie for the picture.")
    else:
        _say(progress, 0.02, "Building a quick preview of your movie from the timeline…")
        st["picture"] = M.build_picture(seq, work, lambda f: _say(
            progress, 0.02 + 0.18 * f, "Building a quick preview of your movie from the "
                                       "timeline…"))
    if cancel and cancel():
        raise StoryError("Stopped.")

    balance_audio(st, seq, progress, cancel)
    clips = st["clips"]

    # stills for Claude
    _say(progress, 0.46, "Taking pictures of every moment…")
    every = 1.0 if dur <= 900 else 2.0
    frames = M.stills(st["picture"], work, every=every)
    sheets = M.contact_sheets(frames, work)
    st["shots"] = [{"start": a, "end": b, "name": (it or {}).get("name", "black")}
                   for a, b, it in P.picture_segments(seq)]
    st["cuts"] = P.shot_changes(seq)

    # Claude plans the sounds
    story = ""
    if st.get("story"):
        try:
            story = read_story(st["story"])
        except StoryError:
            story = ""
    import ss_claude as C
    seq_info = {"name": st["seq_name"], "duration": dur, "shots": st["shots"],
                "audio": [{"start": c["start"], "end": c["end"], "tag": c["tag"],
                           "name": c["name"]} for c in clips]}
    plan = C.plan_sounds(story, seq_info, st["lines"], sheets, settings.get("busy", "normal"),
                         progress=lambda f, m: _say(progress, 0.5 + 0.43 * f, m),
                         music=bool(settings.get("add_music")))
    try:                                     # second read: sounds the story still needs
        plan["cues"] += C.check_missing(story, seq_info, st["lines"], plan["cues"],
                                        bool(settings.get("add_music")),
                                        lambda f, m: _say(progress, 0.95 + 0.04 * f, m))
    except StoryError:
        log("missing-sounds check skipped")
    st["plan_notes"] = plan.get("notes", "")
    st["movie_type"] = plan.get("movie_type", "")
    st["look"] = plan.get("look", "")
    st["cues"] = tidy_cues(plan["cues"], dur, st["cuts"])
    st["stage"] = "planned"
    save(st)
    _say(progress, 1.0, f"Planned {len(st['cues'])} sounds.")
    return st


def balance_audio(st, seq, progress=None, cancel=None, lo=0.2, hi=0.45):
    """Read every audio clip, find where people speak, then clean up and balance the
    narration, music and sound effects (processed copies with the same timing).
    Fills st["clips"], st["speech"], st["lines"], st["audio_map"], st["reset_items"],
    st["levels"]. Shared by "Find sounds" and "Fix the audio only"."""
    work = Path(st["work"])
    dur = seq["duration"]
    span = hi - lo

    def at(f):
        return lo + span * f
    # your audio: every clip, with its volume; narration levelled to sound even
    _say(progress, at(0.0), "Reading the narration and music…")
    clips = M.build_stems(seq, work, lambda f: _say(progress, at(0.4 * f),
                                                   "Reading the narration and music…"))
    _say(progress, at(0.4), "Listening to what is said…")
    import ss_mix
    ss_mix.guess_tags(clips, dur)            # projects without Essential Sound tags
    dia = [c for c in clips if ss_mix.role_of([c]) == "dialog" or c["tag"].startswith("dialog")]
    speech_src = dia or clips
    words = []
    if speech_src:
        tmpwav = work / "speech.wav"
        M.mix(speech_src, dur, tmpwav)
        try:
            words = M.transcribe_raw(M.wav_mono16k(tmpwav), lambda f: _say(
                progress, at(0.4 + 0.4 * f), "Listening to what is said…"))
        except Exception:
            log_error("transcribe")
            words = []
    st["lines"] = M.lines_from_words(words)
    st["speech"] = [[l["s"], l["e"]] for l in st["lines"]]
    if dia and not st["lines"]:              # narration clips mark speech even if unheard
        st["speech"] = [[c["start"], c["end"]] for c in dia]
    if cancel and cancel():
        raise StoryError("Stopped.")

    # clean up and balance your narration and music (processed copies, same timing)
    song_video = not dia
    originals = {}                           # cleaned copies from earlier runs -> originals
    for sf in WORK.glob("*/state.json"):
        try:
            prev = json.loads(sf.read_text(encoding="utf-8-sig"))
            for o, n in (prev.get("audio_map") or {}).items():
                if not ss_mix.is_storysound_sound(n):
                    originals[os.path.normcase(n)] = o
            for n, o in (prev.get("part_files") or {}).items():   # one clip's own file
                originals[os.path.normcase(n)] = o
        except (OSError, ValueError):
            pass
    video_map = {}                           # lip-sync clips with hum: cleaned copies
    try:
        video_map = ss_mix.clean_video_voices(
            clips, work, out_dir(st) / "audio", originals,
            lambda f, m: _say(progress, at(0.8), m))
    except StoryError:
        raise
    except Exception:
        log_error("lip-sync voice cleanup")
    mapping, report, set_levels, per_clip = ss_mix.process(
        clips, st["speech"] if dia else [], work, out_dir(st) / "audio", song_video,
        originals=originals, progress=lambda f, m: _say(progress, at(0.8 + 0.2 * f), m))
    reset = {}                               # track item id -> clip volume (dB) in the copy
    for c in clips:                          # the preview now plays the processed copies
        new = mapping.get(c["file"]) or per_clip.get(c.get("id"))
        if new:
            raw = Path(c["raw"]).with_name(Path(c["raw"]).stem + "_p.f32")
            frames = M.decode_raw(new, raw, c["src_in"], c["end"] - c["start"], c["speed"])
            if frames:
                c.update(raw=str(raw), frames=frames, processed=new)
            c["gain_db"] = c.get("track_db", 0.0)        # effects and clip gain removed
            if c.get("id"):
                reset[c["id"]] = 0.0
        elif c.get("id") in set_levels:
            c["gain_db"] = c.get("track_db", 0.0) + set_levels[c["id"]]
            reset[c["id"]] = set_levels[c["id"]]
        c["level_db"] = 0.0
    st["reset_items"] = reset
    st["clips"] = clips
    st["audio_map"] = dict(mapping, **video_map)
    if video_map:
        n = len(video_map)
        report.insert(min(2, len(report)),
                      f"{n} lip-sync clip{'s' if n != 1 else ''} had the same hum: "
                      f"{'each now uses' if n != 1 else 'it now uses'} a copy with the very same "
                      "picture and the cleaned voice. The clips stay where they are on your "
                      "timeline.")
    st["clip_files"] = per_clip              # track item id -> its own balanced file
    st["part_files"] = {per_clip[c["id"]]: c["file"] for c in clips if c.get("id") in per_clip}
    st["levels"] = report
    st["song_video"] = song_video
    if cancel and cancel():
        raise StoryError("Stopped.")

    return st


def level_clips(clips):
    """Even out the narration: every dialogue clip to the same loudness. Other clips move
    with the narration's average change so your music balance stays as you set it."""
    report = []
    dia = [c for c in clips if c["tag"].startswith("dialog")]
    for c in clips:
        c["loud"] = M.raw_loudness(c["raw"]) + c["gain_db"]
    changes = []
    for c in dia:
        if c["loud"] < -60 or (c["end"] - c["start"]) < 1.0:
            c["level_db"] = 0.0
            continue
        g = max(-10.0, min(10.0, S.DIALOGUE_REF - c["loud"]))
        c["level_db"] = round(g, 1)
        changes.append(g)
    typical = float(np.median(changes)) if changes else 0.0
    for c in clips:
        if not c["tag"].startswith("dialog"):
            c["level_db"] = round(typical, 1)
    if changes:
        spread = max(changes) - min(changes)
        report.append(f"Narration clips varied by {spread:.0f} dB; StoryStudio evened them out.")
        for c in dia:
            if abs(c.get("level_db", 0) - typical) >= 3:
                report.append(f"{fmt_time(c['start'])} {c['name']}: "
                              f"{'louder' if c['level_db'] > typical else 'quieter'} by "
                              f"{abs(c['level_db'] - typical):.0f} dB")
    # music fighting the narration?
    music = [c for c in clips if c["tag"] == "music"]
    if dia and music:
        m = max(c["loud"] + c["level_db"] for c in music)
        if m > S.DIALOGUE_REF - 6:
            report.append("The music is nearly as loud as the narration; consider lowering it "
                          f"about {m - (S.DIALOGUE_REF - 10):.0f} dB under the voice.")
    return report


def tidy_cues(cues, dur, cuts):
    out = []
    for c in cues:
        s, e = max(0.0, float(c["start"])), min(dur, float(c["end"]))
        if e - s < 0.15:
            continue
        if c["kind"] in BEDS:
            s, e = _snap(s, cuts, 1.5), _snap(e, cuts, 1.5)
            if e - s < 2.0:
                continue
        c.update(start=round(s, 3), end=round(e, 3))
        out.append(c)
    # merge back-to-back ambience of the same place
    out.sort(key=lambda c: (c["start"], c["end"]))
    merged = []
    for c in out:
        prev = next((m for m in reversed(merged) if m["kind"] == c["kind"]), None)
        if c["kind"] in BEDS and prev and \
                prev["sound_prompt"].lower() == c["sound_prompt"].lower() and \
                c["start"] - prev["end"] < 1.0:
            prev["end"] = max(prev["end"], c["end"])
            continue
        merged.append(c)
    for i, c in enumerate(merged):
        c.update(id=f"c{i + 1:03d}", keep=True, source="", file="", place_at=c["start"],
                 note="")
    return merged


# wordless reactions: a laugh or a sigh IS a voice, so only words are kept out of them
REACT_WORDS = ("laugh", "giggle", "chuckle", "sigh", "gasp", "snort", "sniffle", "clears his",
               "clears her", "throat", "groan", "murmur of", "hmm")


def is_reaction(c):
    t = f"{c.get('what', '')} {c.get('sound_prompt', '')}".lower()
    return any(w in t for w in REACT_WORDS)


def _snap(t, cuts, reach):
    near = [x for x in cuts if abs(x - t) <= reach]
    return min(near, key=lambda x: abs(x - t)) if near else t


# ---------------------------------------------------------------------------
# 3. make the sounds
# ---------------------------------------------------------------------------
NOT_RIGHT = ("The sound AI couldn't make this one sound right, so it's switched off. Play it "
             "to check, or use “Try a different one”.")


def _seed(c):
    return int(hashlib.md5((c["sound_prompt"] + c.get("seed_salt", "")).encode())
               .hexdigest()[:6], 16)


def sounds_dir(st, fresh=False):
    """Where this run's sound files go: normally "sounds" (a new run replaces the last run's
    files). But while Premiere has an earlier StoryStudio copy of this project open (it
    leaves a .prlock file beside it), that copy is still playing those files, so a new run
    gets its own folder inside "sounds" instead of changing them underneath it."""
    od = out_dir(st)
    if not fresh:
        return Path(st.get("sounds_dir") or od / "sounds")
    d = od / "sounds"
    proj = Path(st["project"])
    base = re.sub(r" \([^()]*\)$", "", proj.stem)
    try:
        in_use = any(f.name.startswith(base + " (") for f in proj.parent.glob("*.prlock"))
    except OSError:
        in_use = False
    if in_use and d.exists() and any(d.glob("*.wav")):
        d = d / time.strftime("run %Y-%m-%d %H%M")
    st["sounds_dir"] = str(d)
    return d


def make_sounds(st, settings, progress=None, cancel=None, ids=None):
    work = Path(st["work"])
    gen = work / "generated"
    gen.mkdir(exist_ok=True)
    sounds = sounds_dir(st, fresh=ids is None or not any(c.get("file") for c in st["cues"]))
    sounds.mkdir(parents=True, exist_ok=True)
    cues = [c for c in st["cues"] if ids is None or c["id"] in ids]
    use_pc = settings.get("use_local", True) and S.local_ready()
    fs_key = settings.get("freesound_key", "").strip() if settings.get("use_freesound") else ""
    # Fal (the storyteller's own account) makes every sound when its key is in Settings:
    # effects by watching the clip, so they land in step; backgrounds from a description
    fal_key = uses_fal(settings)
    use_fal = bool(fal_key)
    make_here = use_pc or use_fal
    if not make_here and not fs_key:
        for c in cues:
            c.update(source="adobe", file="")
        save(st)
        return st
    src = "fal" if use_fal else "pc"
    spent = [0.0]

    def run_ai(jobs, prog=None):
        if use_fal:
            import ss_fal
            errs, used = ss_fal.run_jobs(jobs, fal_key, prog, cancel)
            spent[0] += used["cost"]
            return errs
        return S.run_local(jobs, prog, cancel)

    if use_fal and ids is None:              # Fal: one sound per shot, made by watching it
        cues = st["cues"] = shot_cues(st)
    motion, mfps = M.motion_curve(st["picture"])
    takes = 2 if use_fal else (max(1, int(settings.get("best_of", 3))) if use_pc else 1)
    shot_takes = 2 if int(settings.get("best_of", 3)) > 1 else 1
    hf_key = (settings.get("hf_key") or "").strip() if use_pc and not use_fal else ""
    if ids is None:
        st["fal_cost"] = 0.0
    # plan where each sound comes from
    pc_cues, sao_cues = [], []
    for c in cues:
        why_off = c.get("note", "") if not c.get("keep", True) else ""
        if why_off == NOT_RIGHT:                     # it is about to be made again
            why_off = ""
        c["file"], c["source"], c["note"] = "", "", why_off
        c.pop("text_only", None)
        c.pop("match", None)
        length = c["end"] - c["start"]
        if c["kind"] == "shot":
            if not use_fal:                                # shots are made by Fal only
                c["source"], c["note"] = "adobe", "Add your Fal key in Settings to make this."
                continue
            c["source"] = "fal"
            L = c["end"] - c["start"]
            clip = M.cut_clip(st["picture"], c["start"], L, gen / f"{c['id']}_shot.mp4")
            j = _job(c, gen, str(clip), 0, L, c["start"], takes=shot_takes)
            j["shot"] = True
            pc_cues.append((c, [j]))
        elif c["kind"] in BEDS:
            if use_fal:                                    # one smooth loop, from its words
                c["source"] = "fal"
                pc_cues.append((c, _bed_jobs(c, gen, length, takes, fal=True)))
            elif hf_key:                                   # long natural backgrounds
                c["source"] = "pc"
                sao_cues.append((c, [_job(c, gen, None, 0, min(S.SAO_MAX, max(8, length)),
                                          takes=takes)]))
            elif fs_key:
                c["source"] = "freesound"
            elif use_pc:
                c["source"] = "pc"
                pc_cues.append((c, _bed_jobs(c, gen, length, takes)))
        elif make_here:
            c["source"] = src
            jobs, t, k = [], max(0.0, c["start"] - 0.3), 0
            end = min(st["duration"], c["end"] + 0.4)
            while t < end - 0.2 and k < 6:                # long actions: several pieces
                L = min(S.MAX_GEN, max(1.0, end - t))
                clip = M.cut_clip(st["picture"], t, L, gen / f"{c['id']}_{k}.mp4")
                jobs.append(_job(c, gen, str(clip), k, L, t, takes=takes))
                t += L
                k += 1
            pc_cues.append((c, jobs))
        else:
            c["source"] = "freesound"

    # backgrounds with Stable Audio first; if it can't run, they go to the other AI
    flat_sao = [j for _, js in sao_cues for j in js]
    if flat_sao:
        errs, problem = S.run_sao(flat_sao, hf_key, lambda f, m: _say(progress, 0.03 + 0.2 * f,
                                                                       m), cancel)
        if problem:
            st["notice"] = problem
        bad = {flat_sao[i]["out"] for i in errs}
        for c, js in sao_cues:
            if any(j["out"] in bad for j in js):           # fall back to the video AI
                pc_cues.append((c, _bed_jobs(c, gen, c["end"] - c["start"], takes)))
            else:
                c["_jobs"] = js
                c["engine"] = "stable audio"
    # run the PC jobs in one go (the model loads once)
    flat = [j for _, js in pc_cues for j in js]
    if flat:
        errs = run_ai(flat, lambda f, m: _say(progress, 0.23 + 0.55 * f, m))
        bad = {flat[i]["out"] for i in errs}
        for c, js in pc_cues:
            ok = [j for j in js if j["out"] not in bad]
            if not ok or (c["kind"] not in BEDS and len(ok) < len(js)):
                c["source"] = "freesound" if fs_key and not use_fal else "adobe"
                c["note"] = ("Fal couldn't make this one." if use_fal else
                             "The PC sound AI couldn't make this one.")
            else:
                c["_jobs"] = ok                       # a background can do without a piece
                c["engine"] = "fal" if use_fal else "mmaudio"
    # best of several takes: a judge AI keeps the take that best matches the description
    groups = [{"prompt": c["sound_prompt"], "cands": j["cands"], "out": j["out"],
               "bed": c["kind"] in BEDS}
              for c, js in pc_cues + sao_cues if c.get("_jobs") for j in c["_jobs"]
              if len(j.get("cands") or []) > 1]
    static, best = {}, {}
    if groups:
        static = S.run_pick(groups, lambda f, m: _say(progress, 0.78 + 0.07 * f, m), cancel,
                            scores=best)
    # How well each effect matches its description (the judge's score). Effects are never
    # made again from the words alone: without the picture they lose their sync. One that
    # sounds nothing like it is switched off instead, and the list marks the unsure ones.
    # A shot's sound holds many sounds at once, so its score is only kept for the list.
    for c, js in pc_cues:
        if c["kind"] in BEDS or not c.get("_jobs"):
            continue
        got = [best.get(j["out"]) for j in c["_jobs"]]
        if got and all(g is not None for g in got):
            c["match"] = round(float(np.mean(got)), 3)
            if c["kind"] != "shot" and c["match"] < S.DROP_BELOW:
                c["keep"], c["note"] = False, NOT_RIGHT
    # background pieces that came out as static are left out (no background beats hash);
    # a background with no good piece left gets one more try with fresh pieces
    retry = []
    for c, js in pc_cues + sao_cues:
        if c["kind"] in BEDS and c.get("_jobs"):
            good = [j for j in c["_jobs"] if static.get(j["out"], 0.0) <= 0.3]
            if good:
                c["_jobs"] = good
            else:
                c.pop("_jobs", None)
                retry.append(c)
    if retry and make_here:
        _say(progress, 0.86, f"Remaking {len(retry)} background(s) that came out as static…")
        again = [(c, _bed_jobs(c, gen, c["end"] - c["start"], takes, first=20, fal=use_fal))
                 for c in retry]
        flat2 = [j for _, js in again for j in js]
        errs2 = run_ai(flat2)
        bad2 = {flat2[i]["out"] for i in errs2}
        groups2 = [{"prompt": c["sound_prompt"], "cands": j["cands"], "out": j["out"],
                    "bed": True} for c, js in again for j in js
                   if j["out"] not in bad2 and len(j["cands"]) > 1]
        static2 = S.run_pick(groups2, None, cancel) if groups2 else {}
        for c, js in again:
            for j in js:
                if j["out"] not in bad2 and len(j["cands"]) == 1 and \
                        not Path(j["out"]).exists() and Path(j["cands"][0]).exists():
                    shutil.copyfile(j["cands"][0], j["out"])
            good = [j for j in js if j["out"] not in bad2 and Path(j["out"]).exists()
                    and static2.get(j["out"], 0.0) <= 0.3]
            if good:
                c["_jobs"], c["engine"] = good, ("fal" if use_fal else "mmaudio")
    for c in retry:
        if not c.get("_jobs"):
            c["source"], c["note"] = "adobe", ("Came out sounding like static, so it was left "
                                               "out. Try the Adobe wording if you want it.")
    for c, js in pc_cues + sao_cues:                     # single-take jobs
        for j in c.get("_jobs") or []:
            if len(j.get("cands") or []) == 1 and not Path(j["out"]).exists() and                     Path(j["cands"][0]).exists():
                shutil.copyfile(j["cands"][0], j["out"])
    # Freesound
    used = set()
    fs_cues = [c for c in cues if c["source"] == "freesound"]
    for i, c in enumerate(fs_cues):
        _say(progress, 0.8 + 0.1 * i / max(1, len(fs_cues)),
             f"Finding real recordings on Freesound: {i + 1} of {len(fs_cues)}")
        length = c["end"] - c["start"]
        amb = c["kind"] in BEDS
        hit = S.freesound_search(fs_key, c["search_words"], 10.0 if amb else 0.2,
                                 600.0 if amb else max(3.0, length * 2 + 2), used)
        if not hit and " " in c["search_words"]:
            hit = S.freesound_search(fs_key, " ".join(c["search_words"].split()[:2]),
                                     8.0 if amb else 0.2, 600.0 if amb else 30.0, used)
        if not hit:
            c["source"], c["note"] = "adobe", "No matching recording on Freesound."
            continue
        used.add(hit["id"])
        try:
            p, credit = S.freesound_get(hit, gen / f"{c['id']}_fs.mp3")
            c["_fs"], c["credit"] = str(p), credit
        except Exception:
            log_error("freesound get")
            c["source"], c["note"] = "adobe", "Couldn't download from Freesound."

    # shape, line up, level and save every sound
    for i, c in enumerate(cues):
        _say(progress, 0.9 + 0.1 * i / max(1, len(cues)), "Lining up and levelling sounds…")
        try:
            _shape(st, c, sounds, motion, mfps)
        except Exception:
            log_error(f"shape {c['id']}")
            c["source"], c["file"], c["note"] = "adobe", "", "Something went wrong with this one."
        c.pop("_jobs", None)
        c.pop("_fs", None)
    st["stage"] = "sounds"
    if use_fal:
        st["fal_cost"] = round(float(st.get("fal_cost") or 0.0) + spent[0], 2)
    save(st)
    _say(progress, 1.0, "All sounds are ready to review.")
    return st


def _job(c, gen, video, k, length, at=None, takes=1):
    neg = S.NEGATIVE_MUSIC if c["kind"] == "music" else \
        (S.NEGATIVE_REACTION if is_reaction(c) else S.NEGATIVE)
    return {"video": video, "duration": round(length, 3), "prompt": c["sound_prompt"],
            "cands": [str(gen / f"{c['id']}_{k}_t{t}.wav") for t in range(takes)],
            "negative": neg, "out": str(gen / f"{c['id']}_{k}.wav"),
            "seed": _seed(c) + k, "at": at if at is not None else c["start"]}


SHOT_MIN = 1.2                               # a shot shorter than this joins its neighbour


def shot_cues(st):
    """The sound plan as one job per shot: each shot on the timeline whose planned effects
    are sounds gets one sound, made by watching that whole shot (so the sound maker times
    every sound in it itself). The planned effects become the shot's description; shots with
    nothing to hear (a still under the narration) are skipped. Backgrounds stay as they are.
    The plan's own effects are kept in st["planned_effects"]."""
    if any(c["kind"] == "shot" for c in st["cues"]):
        return st["cues"]
    effects = [c for c in st["cues"] if c["kind"] not in BEDS]
    beds = [c for c in st["cues"] if c["kind"] in BEDS]
    st["planned_effects"] = effects
    spans = []                                   # shots, tiny ones joined to the one before
    for s in st.get("shots") or []:
        a, b = float(s["start"]), float(s["end"])
        if s.get("name") == "black" or b - a < 0.05:
            continue
        if spans and (b - a < SHOT_MIN or spans[-1][1] - spans[-1][0] < SHOT_MIN) and \
                abs(spans[-1][1] - a) < 0.05:
            spans[-1][1] = b
        else:
            spans.append([a, b])
    out = []
    for a, b in spans:
        mine = [c for c in effects if min(b, c["end"]) - max(a, c["start"]) >
                min(0.3, 0.5 * (c["end"] - c["start"]))]
        if not mine or b - a < 1.0:
            continue
        mine.sort(key=lambda c: c["start"])
        prompt = "; ".join(dict.fromkeys(c["sound_prompt"].strip().rstrip(".") for c in mine))
        what = ", ".join(dict.fromkeys(c["what"] for c in mine))
        out.append({"kind": "shot", "start": round(a, 3), "end": round(b, 3),
                    "what": f"Shot {fmt_time(a)}: {what}"[:140], "sound_prompt": prompt[:500],
                    "search_words": "", "suno_prompt": "", "synced": True,
                    "adobe_prompt": "; ".join(c.get("adobe_prompt", "") for c in mine)[:600],
                    "level": "medium" if any(c.get("level") in ("medium", "loud")
                                             for c in mine) else "soft",
                    "includes": [c["what"] for c in mine]})
    cues = sorted(beds + out, key=lambda c: (c["start"], c["end"]))
    for i, c in enumerate(cues):
        c.update(id=f"c{i + 1:03d}", keep=c.get("keep", True), source="", file="",
                 place_at=c["start"], note="")
    log("sound plan as shots:", len(out), "shots with sounds from", len(effects),
        "planned effects;", len(beds), "backgrounds")
    return cues


def uses_fal(settings):
    """The Fal key, when Fal is to make the sounds (a key is saved and it isn't switched
    off); "" otherwise."""
    if settings.get("sound_source", "auto") == "pc":
        return ""
    return str(settings.get("fal_key") or "").strip()


def _bed_jobs(c, gen, length, takes, first=0, fal=False):
    """A long background is made from several different pieces (different seeds) joined
    with long crossfades, so it doesn't audibly repeat the same few seconds over and over.
    Pieces get fewer takes each to keep the time down. With Fal it is one piece that is
    made to loop smoothly."""
    if fal:
        import ss_fal
        j = _job(c, gen, None, first, min(ss_fal.TEXT_MAX, max(4, length)), takes=1)
        j["loop"] = True
        return [j]
    n = max(1, min(6, math.ceil(length / S.MAX_GEN)))
    t = takes if n == 1 else max(1, min(2, takes))
    return [_job(c, gen, None, first + k, min(S.MAX_GEN, max(4, length)), takes=t)
            for k in range(n)]


def _shape(st, c, sounds, motion, mfps):
    length = c["end"] - c["start"]
    if c["kind"] == "shot" and c.get("_jobs"):
        a = S.load48(c["_jobs"][0]["out"])
        n = int(round(length * S.SR))
        a = a[:n] if len(a) >= n else np.concatenate([a, np.zeros((n - len(a), 2), np.float32)])
        a = S.fades(a, 0.004, 0.012)             # no click at the cuts; timing untouched
        place = c["start"]
    elif c["source"] in ("pc", "fal") and c.get("_jobs"):
        bed = c["kind"] in BEDS
        parts = [S.load48(j["out"]) for j in c["_jobs"]]
        parts = [p for p in parts if len(p) > S.SR // 20]   # a piece that didn't come out
        if not parts:
            c["source"], c["file"] = "adobe", ""
            c["note"] = "The PC sound AI couldn't make this one."
            return
        if bed:                              # pieces at the same loudness before joining
            parts = [S.normalise(p, -30.0)[0] if len(p) > S.SR else p for p in parts]
        a = parts[0]
        xf = int((1.5 if bed else 0.05) * S.SR)
        for p in parts[1:]:
            x = min(xf, len(a) // 3, len(p) // 3)
            if x > 8:
                r = np.linspace(0, 1, x, dtype=np.float32)
                if bed:
                    fo, fi = np.cos(r * np.pi / 2)[:, None], np.sin(r * np.pi / 2)[:, None]
                else:
                    fo, fi = (1 - r)[:, None], r[:, None]
                a = np.concatenate([a[:-x], a[-x:] * fo + p[:x] * fi, p[x:]])
            else:
                a = np.concatenate([a, p])
        if c["kind"] in BEDS:
            a, place = _bed(st, c, a)
        else:
            place = c["_jobs"][0]["at"]               # made against the picture: in sync
            a = S.fades(S.trim_tail(a), 0.005, 0.12)
    elif c["source"] == "freesound" and c.get("_fs"):
        a = S.load48(c["_fs"])
        if c["kind"] in BEDS:
            start_at = int(min(len(a) * 0.1, 2 * S.SR))  # skip any handling noise at the top
            a, place = _bed(st, c, a[start_at:])
        else:
            a = a[S.onset(a):]
            a = S.trim_tail(a[: int((length + 1.0) * S.SR)])
            a = S.fades(a, 0.003, min(0.3, len(a) / S.SR / 4))
            place = c["start"]
            if c.get("synced"):
                hit = S.best_hit_time(motion, mfps, c["start"] - 0.4, c["start"] + 0.8)
                if hit is not None:
                    place = hit
    else:
        c["file"] = ""
        return
    if c["kind"] in BEDS:
        import ss_mix
        a = _duck(a, place, ss_mix.merge_regions(st.get("speech", []), gap=ss_mix.HOLD_GAP),
                  dip_db=-3.0, ramp=0.6)
    else:                                    # effects step back a little while someone talks
        a = _duck(a, place, st.get("speech", []), dip_db=-2.0, ramp=0.15)
    if len(a) < S.SR // 20:                  # nothing left to place
        c["source"], c["file"], c["note"] = "adobe", "", "This one came out empty."
        return
    lvl = "background" if c["kind"] in BEDS else (c.get("level") or "soft")
    a, g = S.normalise(a, S.target_lufs(lvl, st.get("movie_type", ""), c["kind"]),
                       None if c["kind"] == "music" else lvl)
    name = safe_name(f"{c['id']} - {c['what']}")[:70] + ".wav"
    path = sounds / name
    S.write_wav(a, path)
    c.update(file=str(path), place_at=round(max(0.0, place), 3),
             length=round(len(a) / S.SR, 3))


def _bed(st, c, a):
    """A background (or music) bed that reaches a little past each scene change on both
    sides, with smooth equal-power fades, so one place crossfades into the next instead of
    dipping to silence at the cut."""
    x = S.BED_OVERLAP.get(c["kind"], 0.8)
    t0 = max(0.0, c["start"] - x)
    t1 = min(st["duration"], c["end"] + x)
    a = S.dehash(S.loop_to(a, int((t1 - t0) * S.SR)))
    fin = 2 * (c["start"] - t0) if c["start"] - t0 > 0.05 else 0.4
    fout = 2 * (t1 - c["end"]) if t1 - c["end"] > 0.05 else 1.0
    return S.fades(a, fin, fout, smooth=True), t0


def _duck(a, place, speech, dip_db=-5.0, ramp=0.35):
    """Ambience dips a little whenever someone speaks (baked into the file)."""
    if not speech:
        return a
    t = place + np.arange(len(a)) / S.SR
    g = np.ones(len(a), dtype=np.float32)
    low = 10 ** (dip_db / 20)
    for s, e in speech:
        if e < place or s > place + len(a) / S.SR:
            continue
        w = np.clip(np.minimum((t - (s - ramp)) / ramp, ((e + ramp) - t) / ramp), 0, 1)
        g = np.minimum(g, 1 - (1 - low) * w.astype(np.float32))
    return a * g[:, None]


# ---------------------------------------------------------------------------
# 4. finish: mix, preview movie, Premiere project copy, sound list
# ---------------------------------------------------------------------------
def fix_audio(st, settings, progress=None, cancel=None):
    """"Fix the audio only": no new sound effects. Cleans and balances the narration,
    music and the sound effects already on the timeline, removes doubled-up Premiere
    effects, double-checks sounds StoryStudio made before (remaking any that came out
    clipped, silent or not matching their description), then writes the project copy and
    a preview."""
    work = Path(st["work"])
    seq = _sequence(st)
    st["mode"] = "audio"
    st["cues"] = []
    _say(progress, 0.01, "Getting the picture ready for the preview…")
    if st.get("export") and Path(st["export"]).exists():
        st["picture"] = st["export"]
    else:
        st["picture"] = M.build_picture(seq, work, lambda f: _say(
            progress, 0.01 + 0.14 * f, "Getting the picture ready for the preview…"))
    if cancel and cancel():
        raise StoryError("Stopped.")
    redone = double_check(st, seq, settings, lambda f, m: _say(progress, 0.15 + 0.2 * f, m),
                          cancel)
    balance_audio(st, seq, progress, cancel, lo=0.35, hi=0.75)
    if redone:
        st["levels"] = [f"Remade {len(redone)} sound effect(s) that didn't come out right: "
                        + ", ".join(redone[:6]) + ("…" if len(redone) > 6 else "")] + \
            st["levels"]
    st["stage"] = "balanced"
    save(st)
    out = finish(st, settings, lambda f, m: _say(progress, 0.75 + 0.25 * f, m), cancel,
                 label="audio fixed")
    return out


def double_check(st, seq, settings, progress=None, cancel=None):
    """Sounds StoryStudio made earlier that are on this timeline: find the run that made
    them, have the judge listen, and remake the ones that are clipped, silent or don't
    match their description. Returns the names of remade sounds."""
    import ss_mix
    _, audio = P.flatten(seq)
    ours = {os.path.normcase(a["file"]) for a in audio
            if a.get("file") and ss_mix.is_storysound_sound(a["file"])}
    if not ours or not settings.get("use_local", True) or not S.local_ready():
        return []
    # which earlier run made each file?
    owners = {}
    for sf in WORK.glob("*/state.json"):
        try:
            prev = load(sf.parent, "cues")
        except (OSError, ValueError):
            continue
        for c in prev.get("cues", []):
            f = os.path.normcase(c.get("file") or "")
            if f in ours:
                owners.setdefault(sf.parent, (prev, []))[1].append(c)
    if not owners:
        return []
    if progress:
        progress(0.05, "Double-checking the sound effects StoryStudio made before…")
    work = Path(st["work"]) / "check"
    work.mkdir(parents=True, exist_ok=True)
    groups, index = [], []
    for wdir, (prev, cues) in owners.items():
        for c in cues:
            groups.append({"prompt": c["sound_prompt"], "cands": [c["file"]],
                           "out": str(work / f"{c['id']}_chk.wav")})
            index.append((wdir, c))
    scores = S.run_score(groups, cancel)
    bad = {}
    for (wdir, c), sc in zip(index, scores):
        a = S.load48(c["file"])
        clipped = len(a) and float((np.abs(a) > 0.99).mean()) > 0.001
        silent = not len(a) or float(np.abs(a).max()) < 0.005
        if clipped or silent or (sc is not None and sc < 0.08):
            bad.setdefault(wdir, []).append(c["id"])
    names = []
    for wdir, ids in bad.items():
        prev = owners[wdir][0]
        for c in prev["cues"]:
            if c["id"] in ids:
                c["seed_salt"] = str(int(c.get("seed_salt") or 0) + 7)
        make_sounds(prev, settings, lambda f, m: progress(0.3 + 0.7 * f, m) if progress
                    else None, cancel, ids=ids)
        names += [c["what"] for c in prev["cues"] if c["id"] in ids]
    return names


def finish(st, settings, progress=None, cancel=None, label="with sound"):
    work = Path(st["work"])
    od = out_dir(st)
    name = safe_name(st["seq_name"])
    if label == "with sound":                # not an earlier "fix the audio" or karaoke run
        st.pop("mode", None)
    kept = [c for c in st["cues"] if c.get("keep") and c.get("file") and Path(c["file"]).exists()]

    # preview mix
    _say(progress, 0.05, "Mixing your movie's sound with the new effects…")
    places = [dict(c, gain_db=c["gain_db"] + c.get("level_db", 0)) for c in st["clips"]]
    for c in kept:
        raw = work / "generated" / f"{c['id']}_placed.f32"
        M.write_raw(raw, S.load48(c["file"]))
        places.append({"raw": str(raw), "start": c["place_at"], "gain_db": 0.0})
    mixwav = work / "mix.wav"
    M.mix(places, st["duration"], mixwav)
    _say(progress, 0.45, "Setting the final loudness (YouTube level)…")
    final = work / "mix_final.wav"
    M.master(str(mixwav), str(final), settings.get("loudness_target", -14.0))
    preview = od / f"{name} - {label} (preview).mp4"
    _say(progress, 0.6, "Making the preview movie…")
    M.mux(st["picture"], str(final), str(preview))
    outputs = {"preview": str(preview), "folder": str(od)}

    # Premiere project copy with the sounds on new tracks
    _say(progress, 0.75, "Adding the sounds to a copy of your Premiere project…")
    try:
        outputs["project"] = write_premiere(st, kept, label)
    except StoryError as ex:
        outputs["project_error"] = str(ex)
    except Exception:
        log_error("write_premiere")
        outputs["project_error"] = ("StoryStudio couldn't add the sounds to the project copy. "
                                    "The sound files are in the StoryStudio folder.")
    if label == "audio fixed":
        # an audio-only run has no new sounds: its notes go in their own file, so the sound
        # list (with all the prompts) from the run that made the sounds is kept
        outputs["list"] = write_audio_notes(st, od / f"{name} - audio fixed notes.txt")
    else:
        outputs["list"] = write_list(st, kept, od / f"{name} - sound list.txt")
    st["outputs"] = outputs
    st["stage"] = "finished"
    save(st)
    _say(progress, 1.0, "Done!")
    return outputs


def preview_moment(st, c, pad=2.5):
    """A few seconds of the movie with this one sound mixed in, to check it."""
    work = Path(st["work"])
    t0 = max(0.0, c.get("place_at", c["start"]) - pad)
    t1 = min(st["duration"], c.get("place_at", c["start"]) + c.get("length", 2.0) + pad)
    places = [dict(x, gain_db=x["gain_db"] + x.get("level_db", 0)) for x in st["clips"]]
    if c.get("file") and Path(c["file"]).exists():
        raw = work / "generated" / f"{c['id']}_placed.f32"
        M.write_raw(raw, S.load48(c["file"]))
        places.append({"raw": str(raw), "start": c["place_at"], "gain_db": 0.0})
    wav = work / "moment.wav"
    M.mix(places, t1 - t0, wav, t0=t0)
    out = work / f"moment_{c['id']}.mp4"
    run([ffmpeg(), "-y", "-v", "error", "-ss", f"{t0:.3f}", "-i", st["picture"], "-i", str(wav),
         "-t", f"{t1 - t0:.3f}", "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-b:a", "192k", str(out)])
    return str(out)


def assign_tracks(cues):
    """Ambience tracks and effect tracks, no two clips overlapping on one track."""
    groups = {"StoryStudio ambience": [c for c in cues if c["kind"] == "ambience"],
              "StoryStudio effects": [c for c in cues if c["kind"] not in BEDS],
              "StoryStudio music": [c for c in cues if c["kind"] == "music"]}
    tracks = []
    for gname, items in groups.items():
        lanes = []
        for c in sorted(items, key=lambda c: c["place_at"]):
            end = c["place_at"] + c.get("length", c["end"] - c["start"])
            for lane in lanes:
                if lane["end"] <= c["place_at"] + 0.01:
                    lane["items"].append(c)
                    lane["end"] = end
                    break
            else:
                lanes.append({"end": end, "items": [c]})
        for i, lane in enumerate(lanes):
            tracks.append((f"{gname} {i + 1}", lane["items"]))
    return tracks


COPY_TAGS = ("with sound", "audio fixed", "karaoke", "feedback")


def copy_path(project, tag):
    """Where a StoryStudio copy of this project goes: "<name> (<tags>).prproj". The tags say
    everything done to it, in order, so a copy made from a copy keeps its history:
    "Song (with sound)" + "audio fixed" -> "Song (with sound, audio fixed)". It replaces the
    earlier copy with the same tags, unless Premiere has that copy open right now (it leaves
    a .prlock file beside it): then a numbered copy instead."""
    real = Path(project)
    stem, tags = real.stem, []
    m = re.search(r" \(([^()]*)\)$", stem)
    if m:
        parts = [re.sub(r" \d+$", "", p.strip()) for p in m.group(1).split(",")]
        if parts and all(p in COPY_TAGS for p in parts):
            stem, tags = stem[:m.start()], parts
    if tag not in tags:
        tags.append(tag)
    label = ", ".join(tags)
    out = real.with_name(f"{stem} ({label}).prproj")
    n = 2
    while out.with_suffix(".prlock").exists() or \
            os.path.normcase(str(out)) == os.path.normcase(str(real)):
        out = real.with_name(f"{stem} ({label} {n}).prproj")
        n += 1
    return out


def write_premiere(st, kept, label="with sound"):
    import ss_prwrite as W
    real = Path(st["project"])
    # built from StoryStudio's own copy (taken when the project was picked), never from the
    # live project you may still be working in
    src = Path(st.get("snapshot") or real)
    out = copy_path(real, label)
    w = W.Writer(src)
    _, ext = w.template_item(".wav")
    tracks = []
    for tname, items in assign_tracks(kept):
        clips = []
        for c in items:
            f = c["file"]
            if ext and ext != ".wav":          # match the kind of file Premiere's copy expects
                f2 = str(Path(f).with_suffix(ext))
                run([ffmpeg(), "-y", "-v", "error", "-i", f, "-b:a", "320k", f2])
                f = f2
            clips.append({"file": f, "start": c["place_at"],
                          "duration": probe(f)["duration"] or c["length"],
                          "name": f"{c['what']}"})
        tracks.append((tname, clips))
    path, _ = W.write_project(src, st["seq_uid"], tracks, out, out_dir(st),
                              repoint=st.get("audio_map") or None,
                              reset=st.get("reset_items") or None,
                              detach=st.get("clip_files") or None)
    return path


def write_audio_notes(st, path):
    lines = [f"StoryStudio – audio fixed for \"{st['seq_name']}\"", "=" * 60, "",
             f"Made from your project as it was at {st.get('snapshot_time', '?')}.", ""]
    lines += ["What was done:"] + [f"  • {x}" for x in st.get("levels", [])] + [""]
    if any(Path(path).parent.glob("* - sound list.txt")):   # only when sounds were added before
        lines += ["The list of sounds with all their prompts is in the “sound list” file from "
                  "the run that made them."]
    Path(path).write_text("\n".join(lines), encoding="utf8")
    return str(path)


def write_list(st, kept, path):
    lines = [f"StoryStudio – sounds for \"{st['seq_name']}\"", "=" * 60, ""]
    if st.get("plan_notes"):
        lines += [st["plan_notes"], ""]
    if st.get("levels"):
        lines += ["Volume:"] + [f"  • {x}" for x in st["levels"]] + [""]
    if st.get("fal_cost"):
        lines += [f"Sounds made with Fal on your account: about ${st['fal_cost']:.2f} (Fal's "
                  "billing page has the exact amount).", ""]
    lines += ["Each sound: time, what it is, where it came from, and wording you can paste "
              "into Adobe's sound effect generator.", ""]
    for c in st["cues"]:
        mark = "✓" if c in kept else "–"
        src = {"pc": "made on this PC", "fal": "made with Fal", "freesound": "Freesound",
               "adobe": "not made yet"}.get(c.get("source"), "")
        lines.append(f"{mark} {fmt_time(c.get('place_at', c['start']))}–{fmt_time(c['end'])}  "
                     f"{c['what']}  ({src})")
        if c.get("includes"):
            lines.append("    In this shot: " + "; ".join(c["includes"]))
        if c["kind"] == "music" and c.get("suno_prompt"):
            lines.append(f"    Suno wording: {c['suno_prompt']}  (about "
                         f"{c['end'] - c['start']:.0f} seconds; the version made here is a "
                         "rough stand-in)")
        lines.append(f"    Adobe wording: {c['adobe_prompt']}")
        if c.get("sound_prompt"):
            lines.append(f"    Wording StoryStudio gave its sound AI: {c['sound_prompt']}")
        if c.get("credit"):
            lines.append(f"    Credit: {c['credit']}")
        lines.append("")
    credits = [c["credit"] for c in kept if c.get("credit")]
    if credits:
        lines += ["Credits to put in your video description:"] + [f"  {x}" for x in credits]
    Path(path).write_text("\n".join(lines), encoding="utf8")
    return str(path)
