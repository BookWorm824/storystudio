"""Cleaning and balancing the storyteller's own audio (narration and music).

Each audio-only source file on the timeline gets a processed copy with EXACTLY the same
length and timing, so the Premiere project copy can simply point at it:
  narration: hum/rumble removed, background hiss reduced, harsh "s" softened, gentle
             compression, then every clip set to the same loudness
  music:     sits a few dB under the narration level, dips further while someone talks
             and comes back up in the gaps; a song video's song is just levelled
Gains are baked per clip into the copy (taking the clip's own Premiere volume into account),
then a limiter keeps peaks safe. Originals are never changed.
"""
import math
import re
import os
from pathlib import Path

import numpy as np

from ss_core import StoryError, ffmpeg, fmt_time, log, log_error, probe, run
import ss_media as M

DIALOGUE_TARGET = -18.0       # every narration clip (LUFS)
MUSIC_ALONE = -24.0           # music when nobody is talking
MUSIC_UNDER = -31.0           # music under narration (about 13 dB below the voice)
HOLD_GAP = 2.5                # music/backgrounds stay down through pauses shorter than this
SONG_TARGET = -16.0           # music videos: the song is the main sound
SFX_TARGET = -30.0            # sound effects already on the timeline (well under the voice)
BG_TARGET = -39.0             # long background sounds (birds, rain, room tone)
SFX_PEAK = -26.0              # loudest moment of an effect / a background, at most
BG_PEAK = -33.0
BG_WORDS = ("ambien", "ambian", "background", "room tone", "atmos", "rain", "wind", "crowd",
            "birds", "forest", "night", "outdoor", "indoor")
AUDIO_ONLY = {".wav", ".mp3", ".aif", ".aiff", ".m4a", ".aac", ".flac", ".ogg"}

CLEAN_MUSIC = "anull"            # music: volume changes only, nothing else touched


def dialogue_chain(src):
    """The narration cleanup, fitted to this recording.

    First the voice is brought to a normal working level, so every later step sees the same
    thing whether the file was recorded quiet or loud. Then the real background noise is
    measured in the quietest moments: noise reduction is used only if there is real hiss,
    gently and set to that measured level (a fixed setting on a quiet recording treats the
    voice itself as hiss, which makes it warble). Clean recordings, like most AI voices,
    get no noise reduction at all."""
    import subprocess
    from ss_core import NOWIN
    r = subprocess.run([ffmpeg(), "-v", "error", "-i", str(src), "-vn", "-t", "900", "-ac", "1",
                        "-ar", "16000", "-f", "f32le", "-"], capture_output=True, **NOWIN)
    a = np.frombuffer(r.stdout[: len(r.stdout) // 4 * 4], dtype=np.float32)
    pre, noise = 0.0, -120.0
    if len(a) > 16000:
        L = M.loudness(a, 16000)
        if L > -69:
            pre = max(-12.0, min(30.0, -20.0 - L))
        fr = a[: len(a) // 800 * 800].reshape(-1, 800)            # 50 ms frames
        rms = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-10)
        live = rms[rms > -100]                                     # ignore pure silence
        if len(live) > 20:
            noise = float(np.percentile(live, 10)) + pre
    parts = [f"volume={pre:.2f}dB", "highpass=f=70", "lowpass=f=16000"]
    if noise > -58:                                    # real hiss: reduce it, gently
        nf = max(-80.0, min(-30.0, noise + 3))
        parts.append(f"afftdn=nr=6:nf={nf:.1f}:tn=0")
    parts += ["deesser=i=0.3:m=0.5:f=0.5",
              "acompressor=threshold=-22dB:ratio=2:attack=15:release=220:makeup=1"]
    log("narration cleanup", Path(src).name, f"pre-gain {pre:+.1f} dB, background {noise:.0f} dB",
        "noise reduction on" if noise > -58 else "no noise reduction needed")
    return ",".join(parts)


def is_storysound_sound(path):
    """A sound effect file made by StoryStudio (or by StorySound, its old name)."""
    p = str(path).replace("/", "\\").lower()
    return "\\storysound\\sounds\\" in p or "\\storystudio\\sounds\\" in p


def already_clean(path):
    """A file StoryStudio made or cleaned before: level it, but don't clean it twice."""
    n = Path(path).name.lower()
    return "(storysound" in n or "(storystudio" in n or is_storysound_sound(path)


NAME_HINTS = (("dialog", ("narrat", "voice over", "voiceover", "dialog", "lipsync", "lip sync",
                            "lip-sync", "talking", "speech")),
              ("music", ("music", "song", "theme", "instrumental", "soundtrack", "score",
                         "suno")),
              ("sfx", ("sfx", "sound effect", "foley", "ambien", "ambian", "whoosh")))


def guess_tags(clips, duration):
    """Clips with no Essential Sound tag (Dialogue / Music / SFX) get a best guess, so a
    project that was never tagged is still treated as a narrated story: by the words in the
    file or clip name, then by the track (a track with many clips that fill a good part of
    the movie is the narration; a long clip is music). Tagged clips are left alone."""
    by_track = {}
    for c in clips:
        if not c.get("tag") and not is_storysound_sound(c["file"]):
            by_track.setdefault(c.get("track"), []).append(c)
    for cs in by_track.values():
        covered = sum(c["end"] - c["start"] for c in cs)
        for c in cs:
            name = f"{c.get('name', '')} {Path(c['file']).name}".lower()
            hint = next((role for role, words in NAME_HINTS if any(w in name for w in words)),
                        None)
            if hint:
                c["tag"] = hint
            elif c["end"] - c["start"] > 45:                 # one long clip: music
                c["tag"] = "music"
            elif len(cs) > 3 and covered >= 0.25 * max(1.0, duration):
                c["tag"] = "dialog"                          # many short clips: the narration
            if c.get("tag"):
                c["guessed"] = True
    return clips


def root_file(path, originals):
    """The recording a cleaned copy was first made from (following every earlier run)."""
    seen = set()
    while True:
        key = os.path.normcase(str(path))
        nxt = (originals or {}).get(key)
        if not nxt or key in seen or not os.path.exists(nxt):
            return path
        seen.add(key)
        path = nxt


def role_of(clips):
    tags = [c["tag"] for c in clips]
    if any(is_storysound_sound(c["file"]) for c in clips):
        return "sfx"
    if any(t.startswith("dialog") for t in tags):
        return "dialog"
    if any(t == "music" for t in tags):
        return "music"
    if any(t in ("sfx", "ambience", "ambient") for t in tags):
        return "sfx"
    # untagged: long clips are probably music, short ones narration
    if sum(c["end"] - c["start"] for c in clips) > 60 and len(clips) <= 3:
        return "music"
    return "dialog" if len(clips) > 3 else "keep"


def merge_regions(regions, gap=0.6):
    out = []
    for s, e in sorted(regions):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _duck_curve(t_seq, speech, depth_db, attack=0.35, release=0.8):
    """0 dB away from speech, depth_db during it, with smooth ramps."""
    d = np.zeros_like(t_seq)
    for s, e in speech:
        w = np.ones_like(t_seq)
        w = np.where(t_seq < s, np.clip(1 - (s - t_seq) / attack, 0, 1), w)
        w = np.where(t_seq > e, np.clip(1 - (t_seq - e) / release, 0, 1), w)
        d = np.minimum(d, depth_db * w)
    return d


def _out_ext(src):
    ext = Path(src).suffix.lower()
    return ext if ext in (".wav", ".mp3", ".m4a", ".aac", ".flac") else ".wav"


def process(clips, speech, work, out_dir, song_video=False, progress=None, originals=None):
    """clips: from ss_media.build_stems (each has file, start, end, gain_db, tag, src_in,
    speed, id, track_db). In StoryStudio's project copy every clip handled here loses its
    Premiere effects and clip gain (StoryStudio's cleanup replaces them), so levels are set
    from the raw file; only the track's fader still applies.
    Returns ({original file: processed file}, report lines, {track item id: volume dB})."""
    import soundfile as sf
    work, out_dir = Path(work), Path(out_dir)
    (work / "clean").mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_file = {}
    for c in clips:
        if Path(c["file"]).suffix.lower() in AUDIO_ONLY:
            by_file.setdefault(c["file"], []).append(c)
    speech = merge_regions(speech)
    held = merge_regions(speech, gap=HOLD_GAP)  # for music/backgrounds: no pumping between lines
    mapping, report, per_clip = {}, [], {}
    dia_levels, music_moves, sfx_moves = [], [], []
    for fi, (src, cs) in enumerate(by_file.items()):
        # a cleaned copy from an earlier run: start again from the original recording
        # (however many runs ago that was), so nothing gets cleaned or lowered twice
        read_src = root_file(src, originals)
        role = "sfx" if is_storysound_sound(read_src) else role_of(cs)
        if role == "keep":
            continue
        clean = read_src != src or not already_clean(src)
        if progress:
            progress(fi / max(1, len(by_file)),
                     f"Cleaning and balancing "
                     f"{ {'dialog': 'the narration', 'music': 'the music'}.get(role, 'the sound effects')}"
                     f" ({Path(src).name})…")
        try:
            info = probe(read_src)
            rate, ch = info.get("sample_rate", 48000), max(1, min(2, info.get("channels", 2)))
            cleaned = work / "clean" / f"src{fi:03d}_clean.wav"
            chain = (dialogue_chain(read_src) if role == "dialog" else CLEAN_MUSIC) if clean \
                else "anull"
            if role == "sfx":
                chain = "highpass=f=20" if clean else "anull"
            run([ffmpeg(), "-y", "-v", "error", "-i", read_src, "-vn", "-af", chain,
                 "-ar", str(rate),
                 "-ac", str(ch), "-c:a", "pcm_f32le", cleaned])
            n_total = sf.info(str(cleaned)).frames
            # gain knots along the SOURCE file's own timeline
            knots_t, knots_g = [], []
            for c in cs:
                s0 = c["src_in"]
                s1 = s0 + (c["end"] - c["start"]) * c["speed"]
                a, _ = sf.read(str(cleaned), start=int(s0 * rate),
                               stop=min(n_total, int(s1 * rate)), dtype="float32",
                               always_2d=True)
                L = M.loudness(_sample(a, rate), rate) if len(a) else -70.0
                if role == "dialog":
                    if L < -60 or (c["end"] - c["start"]) < 0.6:
                        g = 0.0
                    else:
                        g = max(-24.0, min(24.0, DIALOGUE_TARGET - L - c.get("track_db", 0.0)))
                        dia_levels.append((c, L + c["gain_db"]))
                    # a short blend into each clip's level instead of an instant jump
                    e = 0.03 if s1 - s0 > 0.12 else 0.0
                    t = np.array([s0 + e, s1 - e])
                    gg = np.array([g, g])
                elif role == "sfx":
                    long_bg = (c["end"] - c["start"]) >= 8 or any(
                        w in (c["name"] + " " + Path(src).name).lower() for w in BG_WORDS)
                    target = BG_TARGET if long_bg else SFX_TARGET
                    base = 0.0 if L < -60 else max(-30.0, min(18.0, target - L -
                                                              c.get("track_db", 0.0)))
                    if L > -60 and len(a):
                        import ss_sounds
                        mm = ss_sounds.momentary_max(_sample(a, rate), rate)
                        base = min(base, (BG_PEAK if long_bg else SFX_PEAK) - mm -
                                   c.get("track_db", 0.0))
                    t = np.arange(s0, s1 + 0.05, 0.05)
                    t_seq = c["start"] + (t - s0) / c["speed"]
                    gg = base + (_duck_curve(t_seq, held if long_bg else speech,
                                             -5.0 if long_bg else -3.0,
                                             0.5 if long_bg else 0.2,
                                             1.2 if long_bg else 0.4) if speech else 0 * t)
                    sfx_moves.append((c, L, base))
                else:
                    if L < -60:
                        base = 0.0
                    elif song_video or not speech:
                        base = max(-24.0, min(18.0, SONG_TARGET - L - c.get("track_db", 0.0)))
                    else:
                        base = max(-30.0, min(18.0, MUSIC_ALONE - L - c.get("track_db", 0.0)))
                    t = np.arange(s0, s1 + 0.1, 0.1)
                    t_seq = c["start"] + (t - s0) / c["speed"]
                    duck = 0.0 if (song_video or not speech) else MUSIC_UNDER - MUSIC_ALONE
                    gg = base + (_duck_curve(t_seq, held, duck, attack=0.6, release=1.5)
                                 if duck else 0 * t)
                    music_moves.append((c, L + c["gain_db"], base))
                knots_t.append(t)
                knots_g.append(gg)
            ext = _out_ext(src)
            stem = re.sub(r" \((StorySound|StoryStudio)[^)]*\)", "", Path(src).stem)

            def render(kt, kg, label):
                """Bake one volume plan into a copy of the cleaned recording."""
                shaped = work / "clean" / f"src{fi:03d}_shaped.wav"
                with sf.SoundFile(str(cleaned)) as fin,                         sf.SoundFile(str(shaped), "w", samplerate=rate, channels=fin.channels,
                                     subtype="FLOAT") as fout:
                    pos = 0
                    for blk in fin.blocks(blocksize=rate * 20, dtype="float32",
                                          always_2d=True):
                        tt = (pos + np.arange(len(blk))) / rate
                        g = 10 ** (np.interp(tt, kt, kg) / 20)
                        fout.write(blk * g[:, None].astype(np.float32))
                        pos += len(blk)
                # never overwrite a file an earlier "(with sound)" project may be using (or
                # that Premiere has open): pick a fresh name
                out = out_dir / f"{stem} (StoryStudio{label}){ext}"
                n = 2
                while out.exists():
                    out = out_dir / f"{stem} (StoryStudio{label} {n}){ext}"
                    n += 1
                codec = {".wav": ["-c:a", "pcm_s24le"], ".flac": ["-c:a", "flac"],
                         ".mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
                         ".m4a": ["-c:a", "aac", "-b:a", "320k"],
                         ".aac": ["-c:a", "aac", "-b:a", "320k"]}[ext]
                # latency=1: the limiter must not shift the sound later in time
                run([ffmpeg(), "-y", "-v", "error", "-i", shaped, "-af",
                     "alimiter=limit=0.94:attack=3:release=60:level=disabled:latency=1",
                     "-ar", str(rate), *codec, out])
                try:
                    shaped.unlink()
                except OSError:
                    pass
                return str(out)

            # The same stretch of a recording used in more than one place (a song that
            # loops through the movie) needs a different volume plan in each place. One
            # shared file can't hold them all (mixing them made the volume flutter), so
            # each use gets its own file.
            spans = sorted((c["src_in"], c["src_in"] + (c["end"] - c["start"]) * c["speed"])
                           for c in cs)
            shared = any(b0 < a1 - 0.25 for (a0, a1), (b0, b1) in zip(spans, spans[1:]))
            if shared and all(c.get("id") for c in cs):
                for k, (c, t, gg) in enumerate(zip(cs, knots_t, knots_g)):
                    if progress:
                        progress(fi / max(1, len(by_file)),
                                 f"Balancing each use of {Path(src).name} ({k + 1} of "
                                 f"{len(cs)})…")
                    per_clip[c["id"]] = render(t, gg, f" part {k + 1}")
            else:
                order = np.argsort(np.concatenate(knots_t))
                kt = np.concatenate(knots_t)[order]
                kg = np.concatenate(knots_g)[order]
                mapping[src] = render(kt, kg, "")
            shaped = work / "clean" / f"src{fi:03d}_shaped.wav"
            for f in (cleaned, shaped):
                try:
                    f.unlink()
                except OSError:
                    pass
        except Exception as ex:
            if isinstance(ex, StoryError) and str(ex) == "Stopped.":
                raise
            log_error(f"process {src}")                # one bad file doesn't stop the run
            report.append(f"Couldn't clean up {Path(src).name}; it was left as it is.")
    # what changed, in plain words
    if dia_levels:
        vals = [v for _, v in dia_levels]
        spread = max(vals) - min(vals)
        report.insert(0, "Narration cleaned up (hum, hiss and harsh “s” sounds) and every "
                         f"clip set to the same loudness (they differed by up to {spread:.0f} dB).")
        for c, v in dia_levels:
            if abs(v - np.median(vals)) >= 4:
                report.append(f"{fmt_time(c['start'])} {c['name']} was "
                              f"{'quieter' if v < np.median(vals) else 'louder'} than the rest "
                              f"by {abs(v - np.median(vals)):.0f} dB; fixed.")
    if music_moves:
        if song_video or not speech:
            report.append("The song was set to a steady, comfortable level.")
        else:
            report.append("Music now sits under the narration: it dips while someone talks and "
                          "comes back up in the gaps.")
    if sfx_moves:
        report.append(f"{len(sfx_moves)} sound effect clips were set below the narration "
                      "(background sounds lower still) and dip a little while someone talks.")
    # narration inside video files (lipsync clips) can't get a cleaned copy: in the
    # project copy their effects are removed and their clip volume set to match
    set_levels = {}
    for c in clips:
        if c["file"] in mapping or not c.get("id") or c.get("id") in per_clip:
            continue
        if not (c["tag"].startswith("dialog") and
                Path(c["file"]).suffix.lower() not in AUDIO_ONLY):
            continue
        if (c["end"] - c["start"]) < 0.6:
            continue
        L = M.raw_loudness(c["raw"])
        if L < -60:
            continue
        set_levels[c["id"]] = round(max(-24.0, min(30.0, DIALOGUE_TARGET - L -
                                                   c.get("track_db", 0.0))), 2)
    return mapping, report, set_levels, per_clip


def _sample(a, rate, max_sec=120):
    """Long clips: measure 40 evenly spread 3-second pieces (fast, accurate enough)."""
    if len(a) <= rate * max_sec:
        return a
    idx = np.linspace(0, len(a) - rate * 3, 40).astype(int)
    return np.concatenate([a[i:i + rate * 3] for i in idx])
