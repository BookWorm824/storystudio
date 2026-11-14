"""StoryStudio media work: preview movie from a Premiere timeline, audio stems, loudness,
mixing, stills and contact sheets, speech recognition."""
import json
import math
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ss_core import (MODELS, NOWIN, StoryError, ffmpeg, fmt_time, log, log_error, probe, run)
import ss_premiere as P

SR = 48000
PREVIEW_W, PREVIEW_H = 640, 360


# ---------------------------------------------------------------------------
# decoding audio to raw float32 files (mixed later with memmap, so long movies
# never need to fit in memory)
# ---------------------------------------------------------------------------
def _atempo(speed):
    parts, s = [], speed
    while s > 2.0:
        parts.append("atempo=2.0")
        s /= 2.0
    while s < 0.5:
        parts.append("atempo=0.5")
        s /= 0.5
    parts.append(f"atempo={s:.5f}")
    return ",".join(parts)


def decode_raw(src, out_raw, start=0.0, length=None, speed=1.0, channels=2, rate=SR):
    """Decode [start, start+length*speed] of src to float32 interleaved raw. Returns frames."""
    cmd = [ffmpeg(), "-y", "-v", "error"]
    if start > 0.001:
        cmd += ["-ss", f"{start:.4f}"]
    if length is not None:
        cmd += ["-t", f"{length * speed:.4f}"]       # before -i: this much of the source
    cmd += ["-i", str(src)]
    af = ["aresample=async=1:first_pts=0"]
    if abs(speed - 1.0) > 0.001:
        af.append(_atempo(speed))
    cmd += ["-vn", "-af", ",".join(af), "-ac", str(channels), "-ar", str(rate),
            "-f", "f32le", str(out_raw)]
    r = run(cmd, check=False)
    if r.returncode != 0 or not Path(out_raw).exists():
        log("decode failed", src, r.stderr[-800:])
        Path(out_raw).write_bytes(b"")
        return 0
    return Path(out_raw).stat().st_size // (4 * channels)


def raw_view(path, channels=2):
    n = Path(path).stat().st_size // (4 * channels)
    if n == 0:
        return np.zeros((0, channels), dtype=np.float32)
    return np.memmap(path, dtype=np.float32, mode="r", shape=(n, channels))


def write_raw(path, data):
    np.ascontiguousarray(data, dtype=np.float32).tofile(path)


# ---------------------------------------------------------------------------
# loudness (ITU-R BS.1770 K-weighted, gated) in numpy
# ---------------------------------------------------------------------------
def _biquad(x, b, a):
    """Direct-form filter along axis 0 (small loop over samples is too slow, so use
    numpy-only IIR via scipy if present, else a vectorised fallback per channel)."""
    try:
        from scipy.signal import lfilter
        return lfilter(b, a, x, axis=0)
    except ImportError:
        y = np.zeros_like(x)
        x1 = x2 = y1 = y2 = np.zeros(x.shape[1:], dtype=np.float64)
        for i in range(len(x)):
            xi = x[i]
            yi = b[0] * xi + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
            x2, x1, y2, y1 = x1, xi, y1, yi
            y[i] = yi
        return y


def _k_weight(x, rate):
    # pre-filter (high shelf) and RLB high-pass, coefficients for any rate (pyloudnorm form)
    f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    K = math.tan(math.pi * f0 / rate)
    Vh = 10 ** (G / 20)
    Vb = Vh ** 0.4996667741545416
    a0 = 1 + K / Q + K * K
    b = [(Vh + Vb * K / Q + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0]
    a = [1.0, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]
    x = _biquad(x, b, a)
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = math.tan(math.pi * f0 / rate)
    a0 = 1 + K / Q + K * K
    b = [1.0, -2.0, 1.0]
    a = [1.0, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]
    return _biquad(x, b, a)


def loudness(x, rate=SR):
    """Integrated loudness in LUFS of a (n, ch) float array. -70 for silence."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) < rate * 0.1:
        return -70.0
    if len(x) < int(rate * 0.4):
        x = np.concatenate([x, np.zeros((int(rate * 0.4) - len(x), x.shape[1]))])
    # long files: measure on a 4x decimated K-weighted copy is not valid, so cap length
    y = _k_weight(x, rate)
    blk, hop = int(0.4 * rate), int(0.1 * rate)
    n = 1 + (len(y) - blk) // hop
    if n <= 0:
        return -70.0
    sq = (y ** 2).sum(axis=1)
    cs = np.concatenate([[0.0], np.cumsum(sq)])
    z = (cs[np.arange(n) * hop + blk] - cs[np.arange(n) * hop]) / blk
    lk = -0.691 + 10 * np.log10(np.maximum(z, 1e-12))
    z = z[lk > -70]
    if len(z) == 0:
        return -70.0
    rel = -0.691 + 10 * np.log10(z.mean()) - 10
    lk = -0.691 + 10 * np.log10(np.maximum(z, 1e-12))
    z = z[lk > rel]
    if len(z) == 0:
        return -70.0
    return float(-0.691 + 10 * np.log10(z.mean()))


def file_loudness(path):
    """Loudness of any media file's sound, via ffmpeg's ebur128 (fast, no memory)."""
    r = run([ffmpeg(), "-v", "info", "-nostats", "-i", str(path), "-vn", "-af",
             "ebur128=framelog=quiet", "-f", "null", "-"], check=False)
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr or "")
    return float(m[-1]) if m else -70.0


def raw_loudness(path, channels=2, start=0, frames=None):
    v = raw_view(path, channels)
    seg = v[start: start + frames if frames else None]
    if len(seg) > SR * 120:                      # long: sample 40 evenly spaced 3 s pieces
        idx = np.linspace(0, len(seg) - SR * 3, 40).astype(int)
        seg = np.concatenate([seg[i:i + SR * 3] for i in idx])
    return loudness(np.asarray(seg))


# ---------------------------------------------------------------------------
# preview movie from a Premiere sequence
# ---------------------------------------------------------------------------
def _seg_cmd(seg, out, fps):
    a, b, it = seg
    n = max(1, int(round((b - a) * fps)))
    vf = (f"scale={PREVIEW_W}:{PREVIEW_H}:force_original_aspect_ratio=decrease,"
          f"pad={PREVIEW_W}:{PREVIEW_H}:(ow-iw)/2:(oh-ih)/2,setsar=1")
    cmd = [ffmpeg(), "-y", "-v", "error"]
    if it is None:
        cmd += ["-f", "lavfi", "-i", f"color=black:s={PREVIEW_W}x{PREVIEW_H}:r={fps:.6f}"]
    elif Path(it["file"]).suffix.lower() in P.IMAGE_EXTS:
        cmd += ["-loop", "1", "-framerate", f"{fps:.6f}", "-i", it["file"]]
    else:
        src_t = it["src_in"] + (a - it["start"]) * it["speed"]
        if src_t > 0.001:
            cmd += ["-ss", f"{src_t:.4f}"]
        cmd += ["-i", it["file"]]
        if abs(it["speed"] - 1) > 0.001:
            vf = f"setpts=PTS/{it['speed']:.5f}," + vf
        # hold the last frame if the source is shorter than the timeline needs
        vf += ",tpad=stop_mode=clone:stop_duration=30"
    vf += f",fps={fps:.6f},format=yuv420p"
    cmd += ["-an", "-vf", vf, "-frames:v", str(n), "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "30", "-g", str(int(fps * 2)), "-video_track_timescale", "90000",
            "-r", f"{fps:.6f}", str(out)]
    return cmd


def build_picture(seq, work, progress=None):
    """Low-res video of the timeline (top-most clip at each moment), frame-exact."""
    work = Path(work)
    segdir = work / "segments"
    segdir.mkdir(parents=True, exist_ok=True)
    segs = P.picture_segments(seq)
    fps = seq["fps"]
    outs = [segdir / f"seg{i:04d}.mp4" for i in range(len(segs))]
    done = [0]

    def one(i):
        r = subprocess.run(_seg_cmd(segs[i], outs[i], fps), capture_output=True, text=True,
                           encoding="utf8", errors="replace", **NOWIN)
        if r.returncode != 0 or not outs[i].exists():
            log("segment failed", segs[i][2] and segs[i][2]["file"], r.stderr[-600:])
            subprocess.run(_seg_cmd((segs[i][0], segs[i][1], None), outs[i], fps),
                           capture_output=True, **NOWIN)
        done[0] += 1
        if progress:
            progress(done[0] / len(segs))

    with ThreadPoolExecutor(max_workers=max(2, min(6, (os.cpu_count() or 4) // 2))) as ex:
        list(ex.map(one, range(len(segs))))
    lst = work / "segments.txt"
    # paths relative to the list (ffmpeg reads them from the list's folder), so a project name
    # with an apostrophe or other odd characters ("Grandpa's Old Jacket") can't break the list
    lst.write_text("".join(f"file '{o.relative_to(work).as_posix()}'\n" for o in outs),
                   encoding="utf8")
    out = work / "picture.mp4"
    run([ffmpeg(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
         out])
    return str(out)


def build_stems(seq, work, progress=None):
    """Decode every audio clip of the timeline (with its volume and speed) to raw files.
    Returns a list of placements: {raw, start, frames, gain_db, tag, name, file}."""
    work = Path(work) / "audio_clips"
    work.mkdir(parents=True, exist_ok=True)
    _, audio = P.flatten(seq)
    clips = []
    cache = {}
    for i, it in enumerate(audio):
        if not it.get("file") or Path(it["file"]).suffix.lower() in P.IMAGE_EXTS:
            continue
        raw = work / f"clip{i:04d}.f32"
        key = (it["file"], round(it["src_in"], 4), round(it["end"] - it["start"], 4),
               round(it["speed"], 4))
        if key in cache:
            raw = cache[key]
            frames = raw_view(raw).shape[0]
        else:
            frames = decode_raw(it["file"], raw, it["src_in"], it["end"] - it["start"],
                                it["speed"])
            cache[key] = raw
        if frames:
            clips.append({"raw": str(raw), "start": it["start"], "frames": frames,
                          "gain_db": it["gain_db"], "tag": it["tag"] or "",
                          "name": it["name"], "file": it["file"], "track": it["track"],
                          "end": it["end"], "src_in": it["src_in"], "speed": it["speed"],
                          "id": it.get("id"), "track_db": it.get("track_db", 0.0)})
        if progress:
            progress((i + 1) / max(1, len(audio)))
    return clips


# ---------------------------------------------------------------------------
# mixing placements into one wav, block by block
# ---------------------------------------------------------------------------
def mix(placements, duration, out_wav, envelopes=None, block_sec=30.0, t0=0.0):
    """placements: dicts with raw, start (sec), gain_db, optional fade_in/fade_out (sec),
    optional 'env' key naming an entry in envelopes ({name: f(times)->gain}).
    Writes `duration` seconds starting at t0 as a 48 kHz stereo 32-bit float wav."""
    import soundfile as sf
    total = int(math.ceil(duration * SR))
    blk = int(block_sec * SR)
    views = []
    for p in placements:
        v = raw_view(p["raw"])
        if len(v) == 0:
            continue
        s0 = int(round((p["start"] - t0) * SR))
        views.append((p, v, s0))
    with sf.SoundFile(str(out_wav), "w", samplerate=SR, channels=2, subtype="FLOAT") as f:
        for b0 in range(0, total, blk):
            b1 = min(total, b0 + blk)
            acc = np.zeros((b1 - b0, 2), dtype=np.float32)
            for p, v, s0 in views:
                e0 = s0 + len(v)
                lo, hi = max(b0, s0), min(b1, e0)
                if hi <= lo:
                    continue
                seg = np.array(v[lo - s0: hi - s0], dtype=np.float32)
                g = np.full(hi - lo, 10 ** (p.get("gain_db", 0.0) / 20), dtype=np.float32)
                idx = np.arange(lo - s0, hi - s0)
                fi = int(p.get("fade_in", 0) * SR)
                fo = int(p.get("fade_out", 0) * SR)
                if fi > 0:
                    g *= np.clip(idx / fi, 0, 1)
                if fo > 0:
                    g *= np.clip((len(v) - idx) / fo, 0, 1)
                if envelopes and p.get("env") in envelopes:
                    g *= envelopes[p["env"]](t0 + np.arange(lo, hi) / SR).astype(np.float32)
                acc[lo - b0: hi - b0] += seg * g[:, None]
            f.write(acc)
    return str(out_wav)


def master(in_wav, out_wav, target=-14.0, true_peak=-1.5):
    """Two-pass loudness normalisation to the target (YouTube: -14 LUFS)."""
    r = run([ffmpeg(), "-v", "info", "-nostats", "-i", in_wav, "-af",
             f"loudnorm=I={target}:TP={true_peak}:LRA=11:print_format=json", "-f", "null",
             "-"], check=False)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr or "", re.S)
    d = json.loads(m.group(0)) if m else {}
    if "inf" in str(d.get("input_i", "")).lower():      # silence: nothing to set
        af = "anull"
    elif m:
        af = (f"loudnorm=I={target}:TP={true_peak}:LRA=11:measured_I={d['input_i']}:"
              f"measured_TP={d['input_tp']}:measured_LRA={d['input_lra']}:"
              f"measured_thresh={d['input_thresh']}:offset={d['target_offset']}:linear=true")
    else:
        af = f"loudnorm=I={target}:TP={true_peak}:LRA=11"
    run([ffmpeg(), "-y", "-v", "error", "-i", in_wav, "-af", af + ",aresample=48000",
         "-c:a", "pcm_s24le", out_wav])
    return out_wav


def mux(video, audio_wav, out_mp4, copy_video=True):
    cmd = [ffmpeg(), "-y", "-v", "error", "-i", video, "-i", audio_wav, "-map", "0:v:0",
           "-map", "1:a:0"]
    cmd += ["-c:v", "copy"] if copy_video else ["-c:v", "libx264", "-crf", "20"]
    cmd += ["-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", out_mp4]
    run(cmd)
    return out_mp4


# ---------------------------------------------------------------------------
# stills and contact sheets for Claude
# ---------------------------------------------------------------------------
def stills(video, work, every=1.0, width=384):
    out = Path(work) / "stills"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.jpg"):
        f.unlink()
    run([ffmpeg(), "-y", "-v", "error", "-i", video, "-vf",
         f"fps=1/{every},scale={width}:-2", "-q:v", "4",
         str(out).replace("%", "%%") + os.sep + "s%05d.jpg"])
    files = sorted(out.glob("s*.jpg"))
    # frame k (1-based) is the picture at time (k-1)*every + every/2 roughly; use start
    return [((i * every), str(f)) for i, f in enumerate(files)]


def burst(video, t0, t1, n, work, name, width=480):
    """n sharp stills spread evenly from t0 to t1 (a close look at one shot), as
    [(time, path)] with each still's exact time in the movie."""
    out = Path(work) / "close" / name
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.jpg"):
        f.unlink()
    step = max(0.04, (t1 - t0) / max(1, n))
    first = t0 + step / 2
    run([ffmpeg(), "-y", "-v", "error", "-ss", f"{first:.3f}", "-i", video,
         "-t", f"{max(0.05, t1 - t0):.3f}", "-vf", f"fps=1/{step:.4f}:round=near,"
         f"scale={width}:-2", "-q:v", "3", str(out).replace("%", "%%") + os.sep + "f%03d.jpg"])
    files = sorted(out.glob("f*.jpg"))[:n]
    return [(round(first + i * step, 2), str(f)) for i, f in enumerate(files)]


def contact_sheets(frames, work, per_sheet=20, cols=5, label="", fmt=None):
    """Grid images of (time, path) stills with the time printed on each."""
    from PIL import Image, ImageDraw, ImageFont
    out = Path(work) / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("arialbd.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    sheets = []
    for k in range(0, len(frames), per_sheet):
        chunk = frames[k:k + per_sheet]
        ims = [Image.open(p).convert("RGB") for _, p in chunk]
        w, h = ims[0].size
        rows = math.ceil(len(ims) / cols)
        sheet = Image.new("RGB", (cols * w, rows * h), "black")
        d = ImageDraw.Draw(sheet)
        for i, (im, (t, _)) in enumerate(zip(ims, chunk)):
            x, y = (i % cols) * w, (i // cols) * h
            sheet.paste(im, (x, y))
            txt = fmt(t) if fmt else fmt_time(t)
            d.rectangle([x, y, x + 10 + d.textlength(txt, font=font), y + 28], fill="black")
            d.text((x + 5, y + 2), txt, fill="yellow", font=font)
        p = out / f"{label}sheet{k // per_sheet:03d}.jpg"
        sheet.save(p, quality=80)
        sheets.append((chunk[0][0], chunk[-1][0], str(p)))
    return sheets


def cut_clip(video, start, length, out, width=640):
    """A short piece of the picture (for the sound AI to watch)."""
    run([ffmpeg(), "-y", "-v", "error", "-ss", f"{max(0, start):.3f}", "-i", video, "-t",
         f"{length:.3f}", "-an", "-vf", f"scale={width}:-2,fps=25", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "20", str(out)])
    return out


def motion_curve(video, fps_out=24.0):
    """Frame-to-frame change (0..1) of the picture at fps_out, for lining sounds up with
    on-screen action and for finding cuts in exported movies."""
    r = subprocess.run([ffmpeg(), "-v", "error", "-i", video, "-vf",
                        f"fps={fps_out},scale=64:36,format=gray", "-f", "rawvideo", "-"],
                       capture_output=True, **NOWIN)
    a = np.frombuffer(r.stdout, dtype=np.uint8)
    n = len(a) // (64 * 36)
    if n < 2:
        return np.zeros(1), fps_out
    a = a[: n * 64 * 36].reshape(n, 36, 64).astype(np.float32)
    d = np.abs(np.diff(a, axis=0)).mean(axis=(1, 2)) / 255.0
    return np.concatenate([[0.0], d]), fps_out


# ---------------------------------------------------------------------------
# speech recognition (the same Parakeet model CleanView / CleanListen use)
# ---------------------------------------------------------------------------
PARAKEET = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"


def speech_model_dir():
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    for d in (MODELS / PARAKEET, local / "CleanView" / "app" / "models" / PARAKEET,
              local / "CleanListen" / "app" / "models" / PARAKEET,
              local / "CleanView" / "models" / PARAKEET):
        if (d / "tokens.txt").exists():
            return d
    return None


def download_speech_model(progress=None):
    import tarfile
    import urllib.request
    url = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
           f"{PARAKEET}.tar.bz2")
    MODELS.mkdir(parents=True, exist_ok=True)
    tmp = MODELS / (PARAKEET + ".tar.bz2")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            got += len(b)
            if progress and total:
                progress(got / total)
    with tarfile.open(tmp) as t:
        t.extractall(MODELS, filter="data")
    tmp.unlink()
    return MODELS / PARAKEET


def transcribe_raw(mono16k, progress=None, chunk=30.0, overlap=2.0):
    """Words with times from a 16 kHz mono float array."""
    import sherpa_onnx
    d = speech_model_dir() or download_speech_model()
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
        joiner=str(d / "joiner.int8.onnx"), tokens=str(d / "tokens.txt"),
        num_threads=max(1, min(8, os.cpu_count() or 2)), model_type="nemo_transducer")
    R = 16000
    C, O = int(chunk * R), int(overlap * R)
    words, off = [], 0
    while off < len(mono16k):
        seg = mono16k[off: off + C]
        if len(seg) > R * 0.2 and np.abs(seg).max() > 1e-3:
            s = rec.create_stream()
            s.accept_waveform(R, np.ascontiguousarray(seg, dtype=np.float32))
            rec.decode_stream(s)
            res = s.result
            durs = list(res.durations) if res.durations else [0.08] * len(res.tokens)
            first, last = off == 0, off + C >= len(mono16k)
            lo = off / R + (overlap / 2 if not first else -1)
            hi = (off + C) / R - overlap / 2 if not last else 1e12
            cur = []
            for tok, ts, du in zip(res.tokens, res.timestamps, durs):
                st, en = off / R + ts, off / R + ts + max(du, 0.08)
                if tok.startswith(" ") or not cur:
                    cur.append({"t": tok.strip(), "s": st, "e": en})
                else:
                    cur[-1]["t"] += tok
                    cur[-1]["e"] = max(cur[-1]["e"], en)
            words += [w for w in cur if w["t"] and lo <= w["s"] < hi]
        if progress:
            progress(min(1.0, (off + C) / max(1, len(mono16k))))
        if off + C >= len(mono16k):
            break
        off += C - O
    for w in words:
        w["s"], w["e"] = round(w["s"], 2), round(w["e"], 2)
    return words


def wav_mono16k(wav):
    """16 kHz mono float array of a wav (via ffmpeg, low memory)."""
    r = subprocess.run([ffmpeg(), "-v", "error", "-i", str(wav), "-ac", "1", "-ar", "16000",
                        "-f", "f32le", "-"], capture_output=True, **NOWIN)
    return np.frombuffer(r.stdout[: len(r.stdout) // 4 * 4], dtype=np.float32)


def lines_from_words(words, gap=0.8, max_len=14.0):
    """Group words into spoken lines with start/end."""
    lines, cur = [], []
    for w in words:
        if cur and (w["s"] - cur[-1]["e"] > gap or w["e"] - cur[0]["s"] > max_len or
                    cur[-1]["t"][-1:] in ".?!"):
            lines.append(cur)
            cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    return [{"s": l[0]["s"], "e": l[-1]["e"], "text": " ".join(w["t"] for w in l)}
            for l in lines]
