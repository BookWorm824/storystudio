"""Edit feedback: StoryStudio watches your edit and gives plain, kind notes.

    feedback(st, settings, progress, cancel)

1. Exact checks from the timeline itself: flash frames, black gaps, very quick or very long
   shots, a shot used twice, sound that cuts off or jumps, clipping, the overall loudness,
   and for music videos whether the cuts land on the beat.
2. Claude looks at a still of every second with the story / lyrics and the shot list, and
   notes what only eyes can catch: AI glitches (faces and hands changing), a character who
   looks different between shots, pictures that don't match the words, pacing and story.
3. Every note becomes a coloured marker on the timeline in a "(feedback)" copy of the
   project (red = please fix, yellow = idea, blue = beat, green = nicely done), plus a list.
"""
import math
import re
from pathlib import Path

import numpy as np

from ss_core import StoryError, fmt_time, log, log_error, read_story, safe_name
import ss_media as M
import ss_premiere as P

COLOUR = {"fix": "red", "suggest": "yellow", "beat": "blue", "good": "green"}
LABEL = {"fix": "Please fix", "suggest": "Idea", "beat": "Beat", "good": "Nicely done"}


def _note(t, sev, kind, title, note, how="", end=None, source="timeline"):
    return {"time": round(float(t), 3), "end": round(float(end), 3) if end else None,
            "severity": sev, "kind": kind, "title": title, "note": note, "how": how,
            "source": source}


def _frames(sec, fps):
    n = int(round(sec * fps))
    return f"{n} frame" + ("" if n == 1 else "s")


# ---------------------------------------------------------------------------
# 1a. the picture, straight from the timeline
# ---------------------------------------------------------------------------
BRANDED = re.compile(r"intro|outro|title|credit|logo|end ?card", re.I)


def picture_notes(seq, music_video):
    fps = seq["fps"] or 24.0
    frame = 1.0 / fps
    dur = seq["duration"]
    segs = P.picture_segments(seq)
    notes = []
    hold = 7.0 if music_video else 11.0
    shots = [(a, b, it) for a, b, it in segs if it is not None]
    for a, b, it in segs:
        length = b - a
        if it is None:
            if length < frame * 0.5:
                continue
            if a < 0.01 and length < 1.0:
                continue                                  # a moment of black at the start
            if b > dur - 0.01:
                if length > 1.5:
                    notes.append(_note(a, "suggest", "picture", "Ends on black",
                                       f"The picture goes black for the last {length:.1f} "
                                       "seconds while the movie is still running.",
                                       "If that isn't meant as a fade-out, trim the end of the "
                                       "sequence or the sound so they finish together.", b))
                continue
            if length <= 0.5:
                notes.append(_note(a, "fix", "picture", "Black flash between shots",
                                   f"There's a tiny gap with no picture ({_frames(length, fps)}"
                                   "). It shows up as a black flash.",
                                   "Click the empty gap on the timeline and press Delete "
                                   "(Ripple Delete), or drag the next shot to close it.", b))
            else:
                notes.append(_note(a, "suggest", "picture", "Black gap",
                                   f"No picture for {length:.1f} seconds.",
                                   "If it isn't meant as a pause, close the gap or fill it "
                                   "with a shot.", b))
            continue
        name = it.get("name") or "a shot"
        branded = bool(BRANDED.search(name))
        if length < 2.5 * frame:
            notes.append(_note(a, "fix", "picture", "Flash frame",
                               f"Only {_frames(length, fps)} of \"{name}\" shows here, so it "
                               "flashes on screen. This usually happens when two shots on "
                               "different tracks don't quite line up.",
                               "Zoom into the timeline here and either delete the sliver or "
                               "drag the shot above it so it covers the gap.", b))
        elif branded:
            continue                    # her intro / outro / titles are left as they are
        elif length < 0.5 and 0.01 < a and b < dur - 0.01:
            notes.append(_note(a, "suggest", "pacing", "Very quick shot",
                               f"\"{name}\" is only on screen for {length:.2f} seconds, which "
                               "is hard to take in.",
                               "Make it a little longer, or remove it if it isn't needed.", b))
        elif length > hold:
            notes.append(_note(a, "suggest", "pacing", "Long shot",
                               f"\"{name}\" stays on screen for {length:.0f} seconds without a "
                               "cut.",
                               "Keep it if the moment needs time; otherwise split it with a "
                               "close-up or another angle, or add a slow push-in (Scale "
                               "keyframes) so it doesn't feel still.", b))
    # the same piece of a clip used twice
    seen = []
    for a, b, it in shots:
        if not it.get("file") or b - a < 1.0:
            continue
        s0 = it["src_in"]
        s1 = s0 + (b - a) * it.get("speed", 1.0)
        for (pa, pfile, p0, p1) in seen:
            if pfile == it["file"] and min(s1, p1) - max(s0, p0) > 1.0 and a - pa > 2:
                notes.append(_note(a, "suggest", "story", "Shot used twice",
                                   f"This is the same footage as at {fmt_time(pa)}.",
                                   "That can be fine for a chorus, but a different shot here "
                                   "would keep it fresh.", b))
                break
        seen.append((a, it["file"], s0, s1))
    stats = {"shots": len(shots), "average": (sum(b - a for a, b, _ in shots) / len(shots))
             if shots else 0.0, "cuts": [round(a, 3) for a, b, it in segs[1:]
                                         if it is not None]}
    return notes, stats


# ---------------------------------------------------------------------------
# 1b. the sound
# ---------------------------------------------------------------------------
RATE = 16000


def short_loudness(x, rate=RATE, win=0.4, hop=0.1):
    """Loudness (LUFS-like) every hop seconds over win-second windows."""
    y = M._k_weight(np.asarray(x, dtype=np.float64)[:, None], rate)[:, 0]
    sq = np.concatenate([[0.0], np.cumsum(y * y)])
    blk, h = int(win * rate), int(hop * rate)
    n = max(0, 1 + (len(y) - blk) // h)
    idx = np.arange(n) * h
    z = (sq[idx + blk] - sq[idx]) / blk
    return -0.691 + 10 * np.log10(np.maximum(z, 1e-12))


def sound_notes(seq, mono, peak_db, lufs):
    notes = []
    hop = 0.1
    L = short_loudness(mono)
    dur = seq["duration"]

    def at(t):
        return float(L[min(len(L) - 1, max(0, int(t / hop)))]) if len(L) else -70.0

    def mean(t0, t1):
        a, b = max(0, int(t0 / hop)), min(len(L), int(t1 / hop))
        return float(np.mean(L[a:b])) if b > a else -70.0
    # sound that drops out in the middle
    loud = np.where(L > -50)[0]
    if len(loud):
        first, last = loud[0], loud[-1]
        quiet = L[first:last] < -55
        i = 0
        while i < len(quiet):
            if quiet[i]:
                j = i
                while j < len(quiet) and quiet[j]:
                    j += 1
                if (j - i) * hop >= 1.5:
                    t = (first + i) * hop
                    notes.append(_note(t, "suggest", "sound", "Sound drops out",
                                       f"Everything goes silent for {(j - i) * hop:.1f} "
                                       "seconds.",
                                       "If the silence isn't on purpose, let the music or "
                                       "background sound carry on underneath.",
                                       t + (j - i) * hop))
                i = j
            else:
                i += 1
    # clips that start or stop suddenly
    _, audio = P.flatten(seq)
    edges = []
    for it in audio:
        if it.get("file"):
            edges.append((it["start"], "start", it))
            edges.append((it["end"], "end", it))
    edges.sort(key=lambda e: e[0])
    done = []
    for t, kind, it in edges:
        if t < 0.3 or any(abs(t - d) < 1.0 for d in done):
            continue
        before, after = mean(t - 0.6, t - 0.05), mean(t + 0.05, t + 0.6)
        name = it.get("name") or "a sound"
        if kind == "end" and before > -32 and before - after > 14:
            last = t > dur - 0.5 or t >= max(x["end"] for x in audio) - 0.05
            notes.append(_note(t, "suggest", "sound",
                               "Music stops suddenly" if last else "Sound cuts off",
                               f"\"{name}\" stops abruptly here.",
                               "Add a short fade: right-click the end of the audio clip and "
                               "choose Apply Default Transitions (Constant Power), or drag a "
                               "volume keyframe down over the last second.", None))
            done.append(t)
        elif kind == "start" and after > -30 and after - before > 16 and t < dur - 0.5:
            notes.append(_note(t, "suggest", "sound", "Sound jumps in",
                               f"\"{name}\" starts suddenly and loudly.",
                               "Add a short fade-in (Constant Power) at the start of the clip "
                               "so it doesn't make people jump.", None))
            done.append(t)
    # too loud / too quiet overall
    if peak_db > -0.3:
        t = float(np.argmax(np.abs(mono))) / RATE
        notes.append(_note(t, "fix", "sound", "Sound is clipping",
                           "The sound hits the very top of the meter here and may crackle.",
                           "Turn this part down a little, or use StoryStudio's \"Fix the "
                           "audio\" to balance everything."))
    if lufs < -24:
        notes.append(_note(0.0, "suggest", "sound", "Quiet overall",
                           f"The whole movie is quiet (about {lufs:.0f} LUFS; YouTube plays "
                           "around -14).",
                           "Use StoryStudio's \"Fix the audio\", or raise everything evenly "
                           "before exporting."))
    return notes


# ---------------------------------------------------------------------------
# 1c. the beat (music videos)
# ---------------------------------------------------------------------------
def beats(mono, rate=RATE):
    """Beat times (s) and the beat length, from the song: onset strength, tempo by
    autocorrelation, then dynamic-programming beat tracking."""
    hop, nfft = 256, 1024
    if len(mono) < nfft * 4:
        return [], 0.5
    n = 1 + (len(mono) - nfft) // hop
    win = np.hanning(nfft).astype(np.float32)
    flux = np.zeros(n, dtype=np.float32)
    prev = None
    for k in range(0, n, 2048):                       # in blocks, to keep memory low
        idx = np.arange(k, min(n, k + 2048))
        frames = np.stack([mono[i * hop:i * hop + nfft] for i in idx]) * win
        mag = np.log1p(30 * np.abs(np.fft.rfft(frames, axis=1)))
        if prev is not None:
            mag = np.vstack([prev, mag])
            d = np.maximum(0, np.diff(mag, axis=0)).sum(1)
        else:
            d = np.concatenate([[0], np.maximum(0, np.diff(mag, axis=0)).sum(1)])
        flux[idx] = d[-len(idx):]
        prev = mag[-1:]
    fr = rate / hop                                   # onset frames per second
    flux -= np.convolve(flux, np.ones(int(fr)) / int(fr), mode="same")   # remove the trend
    flux = np.maximum(flux, 0)
    flux /= flux.std() + 1e-9
    # tempo: autocorrelation, leaning towards ~110 bpm
    seg = flux[: int(fr * 120)]
    ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
    lags = np.arange(int(fr * 60 / 180), int(fr * 60 / 60))
    w = np.exp(-0.5 * (np.log2(60 * fr / lags / 110.0) / 0.9) ** 2)
    period = int(lags[np.argmax(ac[lags] * w)])
    # dynamic programming beat tracking (Ellis 2007)
    score = flux.astype(np.float64).copy()
    back = -np.ones(len(flux), dtype=int)
    lo, hi = int(period / 2), int(period * 2)
    for t in range(lo, len(flux)):
        a, b = max(0, t - hi), t - lo
        if b <= a:
            continue
        prevs = np.arange(a, b)
        pen = -100 * (np.log((t - prevs) / period)) ** 2
        cand = score[a:b] + pen
        j = int(np.argmax(cand))
        if cand[j] > 0:
            score[t] += cand[j]
            back[t] = prevs[j]
    t = int(np.argmax(score[-hi:])) + len(score) - hi
    out = []
    while t >= 0:
        out.append(t / fr)
        t = back[t]
    return sorted(out), period / fr


def beat_notes(cuts, beat_times, beat_len, fps, has_music_from):
    if len(beat_times) < 8:
        return [], None
    bt = np.array(beat_times)
    tol = max(2.0 / fps, 0.07)
    on, notes = 0, []
    judged = [c for c in cuts if c >= has_music_from and c <= bt[-1] + beat_len]
    for c in judged:
        i = int(np.argmin(np.abs(bt - c)))
        d = c - bt[i]
        if abs(d) <= tol:
            on += 1
            continue
        if abs(d) > beat_len * 0.3:
            continue                    # nearer half-way between beats: likely on purpose
        n = int(round(abs(d) * fps))
        notes.append(_note(c, "beat", "music", "Cut is off the beat",
                           f"This cut is {n} frame{'s' if n != 1 else ''} "
                           f"{'after' if d > 0 else 'before'} the beat.",
                           f"Use the Rolling Edit tool (N) and drag this cut {n} frame"
                           f"{'s' if n != 1 else ''} {'left' if d > 0 else 'right'} so the "
                           "new shot lands on the beat."))
    summary = {"on_beat": on, "cuts": len(judged), "bpm": round(60.0 / beat_len)}
    return notes, summary


# ---------------------------------------------------------------------------
# 2. Claude watches it
# ---------------------------------------------------------------------------
NOTE = {
    "type": "object",
    "properties": {
        "time": {"type": "number", "description": "seconds from the start of the movie"},
        "end": {"type": "number", "description": "seconds; same as time for a single moment"},
        "severity": {"type": "string", "enum": ["fix", "suggest", "good"]},
        "kind": {"type": "string", "enum": ["ai_glitch", "continuity", "story", "pacing",
                                            "picture", "music", "titles"]},
        "title": {"type": "string", "description": "2 to 6 plain words"},
        "note": {"type": "string", "description": "what you noticed, one or two plain "
                                                  "sentences"},
        "how": {"type": "string", "description": "how to fix it in Premiere (or how to "
                                                 "regenerate the shot), plainly; empty for "
                                                 "'good' notes"},
    },
    "required": ["time", "end", "severity", "kind", "title", "note", "how"],
    "additionalProperties": False,
}
SCHEMA = {
    "type": "object",
    "properties": {
        "notes": {"type": "array", "items": NOTE},
        "overall": {"type": "string", "description": "2-3 warm, honest sentences about the "
                                                     "whole edit"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "top_fixes": {"type": "array", "items": {"type": "string"},
                      "description": "the 1-3 changes that would help most, plainly"},
    },
    "required": ["notes", "overall", "strengths", "top_fixes"],
    "additionalProperties": False,
}
SYSTEM = """You are a kind, experienced film editor giving notes to a storyteller who makes \
children's stories, short movies and music videos with AI-generated video in Adobe Premiere. \
She is not technical: write in plain, warm, everyday words, no jargon. Be specific and honest; \
every note must point at a real moment you can see in the stills, with its time.

Look for what only eyes can catch (the timeline facts are checked separately; don't repeat \
them):
- AI glitches: faces or hands that change or melt, extra fingers or limbs, objects morphing, \
warped or nonsense text and letters, characters popping in or out.
- Continuity: the same character looking different between shots (hair, clothes, age, colours), \
props or settings that change, day turning to night by accident.
- The picture matching the words: narration or lyrics talking about one thing while the picture \
shows another; a shot that would fit better somewhere else.
- Story and pacing: is it clear what is happening and who is who? does the opening grab \
attention? does the ending land? a part that drags or rushes; two very similar shots back to \
back.
- Picture: shots much brighter/darker or a different colour from their neighbours, heads cut \
off, the subject tiny or off to the side, titles that are hard to read.
- Music videos: shots that match the feel of the music and lyrics, variety, big moments on the \
big musical moments.
- Studio rules (always "fix"): any religious symbol (crosses, halos, church icons...), Jesus or \
God shown as a person or figure (He may only appear as a warm light), and any Christmas, \
Halloween, Easter or St. Patrick's Day imagery.

Severity: "fix" = a clear mistake viewers will notice; "suggest" = an idea that would make it \
better; "good" = a genuinely strong moment (give 2 to 4 of these so she knows what is working). \
Skip nit-picks; quality over quantity (usually 6 to 20 notes for a few minutes). In "how", give \
a concrete step: which shot to trim, swap, reorder or regenerate, and what to ask the video AI \
for when regenerating (e.g. "regenerate with: same girl, short brown hair, blue hoodie")."""


def _shot_list(shots, lo, hi):
    return "\n".join(f"{fmt_time(a)}-{fmt_time(b)} ({a:.2f}-{b:.2f}s) {name}"
                     for a, b, name in shots if b > lo and a < hi) or "(none)"


def claude_notes(st, kind, story, lines, shots, facts, sheets, progress=None,
                 per_request=8):
    import ss_claude as C
    groups = [sheets[i:i + per_request] for i in range(0, len(sheets), per_request)] or [[]]
    dur = st["duration"]
    out = {"notes": [], "overall": "", "strengths": [], "top_fixes": []}
    words = "\n".join(f"{fmt_time(l['s'])}-{fmt_time(l['e'])}: {l['text']}" for l in lines)
    story_text = (story or "").strip()[:20000]
    fact_text = "\n".join(f"{fmt_time(n['time'])} {n['title']}" for n in facts) or "(none)"
    for gi, grp in enumerate(groups):
        lo = grp[0][0] if grp else 0.0
        hi = groups[gi + 1][0][0] if gi + 1 < len(groups) else dur
        if progress:
            progress(gi / len(groups), f"Claude is watching {fmt_time(lo)} to {fmt_time(hi)}…")
        prompt = f"""This is a {"music video" if kind == "music" else "story / short movie"}: \
"{st['seq_name']}", {dur:.1f} seconds long.
This request covers {lo:.1f}s to {hi:.1f}s{f' (part {gi + 1} of {len(groups)})' if len(groups) > 1 else ''}.

{"THE LYRICS" if kind == "music" else "THE STORY (her script)"}:
{story_text or "(not given; use what is said or sung below)"}

SHOTS ON THE TIMELINE (exact times; file names are often just numbers):
{_shot_list(shots, lo, hi)}

WHAT IS SAID / SUNG (speech recognition, may mishear words and names):
{words or "(nothing heard)"}

ALREADY FOUND BY THE TIMELINE CHECK (don't repeat these):
{fact_text}

The pictures are stills of this part, one every {st.get('_still_every', 1.0):g} second(s), with \
the time in yellow. Give your notes for {lo:.1f}s to {hi:.1f}s, times in seconds from the \
start of the movie."""
        images = [(f"Stills {fmt_time(a)} to {fmt_time(b)}:", p) for a, b, p in grp]
        res = C.ask(prompt, SCHEMA, system=SYSTEM, images=images)
        for n in res.get("notes", []):
            t = float(n.get("time", 0))
            if not (lo - 2 <= t <= hi + 2):
                continue
            e = float(n.get("end") or t)
            out["notes"].append(_note(t, n.get("severity", "suggest"), n.get("kind", "story"),
                                      n.get("title", "").strip() or "Note",
                                      n.get("note", "").strip(), n.get("how", "").strip(),
                                      e if e > t + 0.3 else None, source="ai"))
        if gi == 0:
            out.update(overall=res.get("overall", ""), strengths=res.get("strengths", []),
                       top_fixes=res.get("top_fixes", []))
        else:
            out["strengths"] += res.get("strengths", [])[:2]
            out["top_fixes"] += res.get("top_fixes", [])[:2]
    if len(groups) > 1:                               # one overall view of the whole movie
        try:
            listing = "\n".join(f"{fmt_time(n['time'])} [{n['severity']}] {n['title']}: "
                                f"{n['note']}" for n in out["notes"])
            res = C.ask(f"""Notes on the edit of "{st['seq_name']}" ({dur:.0f} s), \
made part by part:\n{listing}\n\nTimeline check found:\n{fact_text}\n\nWrite the overall view \
of the whole movie. Return notes as an empty list.""", SCHEMA, system=SYSTEM)
            out.update(overall=res.get("overall", out["overall"]),
                       strengths=res.get("strengths", out["strengths"])[:4],
                       top_fixes=res.get("top_fixes", out["top_fixes"])[:3])
        except StoryError:
            pass
    return out


# ---------------------------------------------------------------------------
# 3. all of it
# ---------------------------------------------------------------------------
def guess_kind(seq, clips):
    """'music' when the sound is mostly one song with no narration clips."""
    dur = seq["duration"] or 1
    import ss_mix
    if any(ss_mix.role_of([c]) == "dialog" or c["tag"].startswith("dialog") for c in clips):
        return "story"
    music = sum(c["end"] - c["start"] for c in clips
                if c["tag"] == "music" or ss_mix.role_of([c]) == "music")
    return "music" if music > 0.6 * dur else "story"


def feedback(st, settings, progress=None, cancel=None, kind="auto", use_claude=True):
    import storysound as E
    say = progress or (lambda f, m: None)

    def stop():
        if cancel and cancel():
            raise StoryError("Stopped.")
    work = Path(st["work"]) / "feedback"
    work.mkdir(parents=True, exist_ok=True)
    seq = E._sequence(st)
    st.setdefault("width", seq["width"])
    st.setdefault("height", seq["height"])
    dur = seq["duration"]
    # the picture
    if not st.get("picture") or not Path(st["picture"]).exists():
        if st.get("export") and Path(st["export"]).exists():
            st["picture"] = st["export"]
        else:
            say(0.01, "Getting the picture ready…")
            st["picture"] = M.build_picture(seq, Path(st["work"]), lambda f: say(
                0.01 + 0.15 * f, "Getting the picture ready…"))
    stop()
    # the sound exactly as the timeline has it
    say(0.17, "Listening to the sound…")
    clips = M.build_stems(seq, work, lambda f: say(0.17 + 0.08 * f, "Listening to the sound…"))
    mixwav = work / "timeline_mix.wav"
    M.mix([dict(c, gain_db=c.get("gain_db", 0.0) + c.get("track_db", 0.0)) for c in clips],
          dur, mixwav)
    mono = M.wav_mono16k(mixwav)
    stop()
    if kind not in ("music", "story"):
        kind = guess_kind(seq, clips)
    st["feedback_kind"] = kind
    # 1. the exact checks
    say(0.27, "Checking every cut on the timeline…")
    notes, stats = picture_notes(seq, kind == "music")
    peak = 20 * math.log10(max(1e-9, float(np.abs(mono).max()) if len(mono) else 1e-9))
    lufs = M.loudness(mono[:, None], RATE) if len(mono) else -70.0
    try:
        notes += sound_notes(seq, mono, peak, lufs)
    except Exception:
        log_error("feedback sound notes")
    beat_sum = None
    if kind == "music":
        say(0.32, "Finding the beat of the song…")
        try:
            loud = np.where(short_loudness(mono) > -45)[0]
            music_from = loud[0] * 0.1 if len(loud) else 0.0
            bt, blen = beats(mono)
            bn, beat_sum = beat_notes(stats["cuts"], bt, blen, seq["fps"] or 24.0, music_from)
            if beat_sum and beat_sum["cuts"] and beat_sum["on_beat"] / beat_sum["cuts"] < 0.25:
                bn = []          # cuts were placed freely, not to the beat: one note, not 40
                notes.append(_note(music_from, "suggest", "music", "Cutting to the beat",
                                   f"Most cuts don't land on the beat of the song (about "
                                   f"{beat_sum['bpm']} beats a minute). Cutting on the beat "
                                   "makes a music video feel tight and musical.",
                                   "Turn on snapping, add markers on the beats (press M while "
                                   "the song plays), and line your cuts up with them."))
            notes += bn[:30]
            st["beats"] = [round(b, 3) for b in bt]
        except Exception:
            log_error("feedback beats")
    stop()
    # what is said or sung
    say(0.36, "Listening to the words…")
    lines = []
    try:
        lines = M.lines_from_words(M.transcribe_raw(mono, lambda f: say(
            0.36 + 0.1 * f, "Listening to the words…")))
    except Exception:
        log_error("feedback transcribe")
    story = ""
    if kind == "music" and st.get("lyrics_text"):
        story = st["lyrics_text"]
    elif st.get("story"):
        try:
            story = read_story(st["story"])
        except StoryError:
            story = ""
    stop()
    # 2. Claude watches
    summary = {"overall": "", "strengths": [], "top_fixes": []}
    import ss_claude as C
    by = C.ai_name() if use_claude else ""
    if use_claude and not C.provider():
        use_claude = False
        summary["overall"] = ("Claude or ChatGPT isn't set up on this computer, so these notes "
                              "come from the timeline checks only. Open Settings to set up "
                              "ChatGPT.")
    shots = [(a, b, (it or {}).get("name", "black")) for a, b, it in P.picture_segments(seq)]
    if use_claude:
        say(0.47, "Taking a picture of every second…")
        every = 1.0 if dur <= 420 else 2.0
        st["_still_every"] = every
        frames = M.stills(st["picture"], work, every=every)
        sheets = M.contact_sheets(frames, work, label="fb")
        try:
            res = claude_notes(st, kind, story, lines, shots, notes, sheets,
                               lambda f, m: say(0.5 + 0.4 * f, m.replace("Claude", by)))
            notes += _dedupe(res["notes"], notes)
            summary = {k: res[k] for k in ("overall", "strengths", "top_fixes")}
        except StoryError as ex:
            log("feedback: Claude skipped", ex)
            summary["overall"] = (f"{by} couldn't watch the movie this time ({ex}) These "
                                  "notes come from the timeline checks only.")
    stop()
    notes.sort(key=lambda n: n["time"])
    for i, n in enumerate(notes):
        n["id"] = i + 1
        n["keep"] = True
    st["feedback"] = {"kind": kind, "notes": notes, "summary": summary, "by": by, "stats": {
        "shots": stats["shots"], "average": round(stats["average"], 2),
        "lufs": round(lufs, 1), "peak": round(peak, 1), "beat": beat_sum}}
    # a preview with sound, for clicking through the notes
    say(0.92, "Making a preview to click through…")
    prev = work / "feedback preview.mp4"
    try:
        M.mux(st["picture"], str(mixwav), str(prev))
        st["feedback_preview"] = str(prev)
    except Exception:
        log_error("feedback preview")
        st["feedback_preview"] = st["picture"]
    out = write_outputs(st)
    E.save(st)
    say(1.0, f"Done: {len(notes)} notes.")
    return out


def _dedupe(new, old):
    keep = []
    for n in new:
        if any(abs(n["time"] - o["time"]) < 1.0 and n["kind"] == o["kind"] for o in old + keep):
            continue
        keep.append(n)
    return keep


def write_outputs(st):
    """The notes list and the "(feedback)" project copy with a marker for every note
    that is switched on."""
    import storysound as E
    import ss_prwrite as W
    fb = st["feedback"]
    od = E.out_dir(st)
    name = safe_name(st["seq_name"])
    notes = [n for n in fb["notes"] if n.get("keep", True)]
    out = {"folder": str(od)}
    lst = od / f"{name} - edit feedback.txt"
    lst.write_text(notes_text(st, notes), encoding="utf8")
    out["list"] = str(lst)
    real = Path(st["project"])
    import storysound as _E
    proj = _E.copy_path(real, "feedback")
    markers = [{"start": x["time"], "end": x.get("end"), "colour": COLOUR[x["severity"]],
                "name": f"{LABEL[x['severity']]}: {x['title']}",
                "comment": x["note"] + (f"\n\nHow: {x['how']}" if x.get("how") else "")}
               for x in notes]
    try:
        out["project"] = W.write_markers(st.get("snapshot") or st["project"], st["seq_uid"],
                                         markers, proj)
    except StoryError as ex:
        out["project_error"] = str(ex)
    except Exception:
        log_error("feedback markers")
        out["project_error"] = ("StoryStudio couldn't put the notes on the timeline, but the "
                                "notes list is in the StoryStudio folder.")
    st["outputs"] = out
    return out


def notes_text(st, notes):
    fb = st["feedback"]
    s, stt = fb["summary"], fb["stats"]
    lines = [f"Edit feedback for \"{st['seq_name']}\"", ""]
    if s.get("overall"):
        lines += [s["overall"], ""]
    if s.get("strengths"):
        lines += ["What's working:"] + [f"  + {x}" for x in s["strengths"]] + [""]
    if s.get("top_fixes"):
        lines += ["What would help most:"] + [f"  - {x}" for x in s["top_fixes"]] + [""]
    lines.append(f"{stt['shots']} shots, {stt['average']:.1f} seconds each on average. "
                 f"Loudness about {stt['lufs']:.0f} LUFS.")
    if stt.get("beat"):
        b = stt["beat"]
        lines.append(f"Song: about {b['bpm']} beats a minute; {b['on_beat']} of {b['cuts']} "
                     "cuts land on the beat.")
    lines += ["", "Notes (they are also markers on the timeline in the \"(feedback)\" copy: "
              "red = please fix, yellow = idea, blue = beat, green = nicely done):", ""]
    for n in notes:
        lines.append(f"{fmt_time(n['time'])}  [{LABEL[n['severity']]}] {n['title']}")
        lines.append(f"        {n['note']}")
        if n.get("how"):
            lines.append(f"        How: {n['how']}")
        lines.append("")
    return "\n".join(lines)
