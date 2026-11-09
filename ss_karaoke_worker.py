"""Times every word of a song's lyrics to the singing.

Run by StoryStudio:  python ss_karaoke_worker.py job.json
job.json: {"models_dir": ..., "audio": song wav (any rate), "words": ["THERE'S", "A", ...],
           "out": result json path}
1. Demucs separates the singing voice from the music (much more reliable timing).
2. A speech model (wav2vec2) hears letters in the voice, and the lyrics are lined up with it
   by forced alignment: every word gets a start and end time.
Prints JSON lines: {"event": "loading"|"separating"|"aligning"|"done"|"fatal", ...}
"""
import json
import os
import subprocess
import sys
from pathlib import Path


def say(**kw):
    print(json.dumps(kw), flush=True)


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf8"))
    home = Path(spec["models_dir"])
    os.environ.setdefault("HF_HOME", str(home / "hf"))
    os.environ.setdefault("TORCH_HOME", str(home / "torch"))
    import numpy as np
    import torch
    ffmpeg = spec["ffmpeg"]
    device = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 2.5 * 2 ** 30 \
        else "cpu"

    def load(path, rate, ch):
        r = subprocess.run([ffmpeg, "-v", "error", "-i", path, "-ac", str(ch), "-ar", str(rate),
                            "-f", "f32le", "-"], capture_output=True)
        a = np.frombuffer(r.stdout[: len(r.stdout) // (4 * ch) * 4 * ch], dtype=np.float32)
        return a.reshape(-1, ch)

    # 1. the singing voice on its own
    say(event="separating", device=device)
    vocals16 = None
    try:
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
        model = get_model("htdemucs").to(device).eval()
        mix = load(spec["audio"], model.samplerate, 2)
        wav = torch.from_numpy(mix.T.copy()).float()
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)
        with torch.no_grad():
            src = apply_model(model, wav[None].to(device), device=device, split=True,
                              overlap=0.25, progress=False)[0]
        src = src * (ref.std() + 1e-8) + ref.mean()
        v = src[model.sources.index("vocals")].mean(0).cpu().numpy()
        # to 16 kHz for the speech model
        n = int(len(v) * 16000 / model.samplerate)
        vocals16 = np.interp(np.linspace(0, len(v) - 1, n), np.arange(len(v)), v).astype(np.float32)
        del model, src
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception as ex:
        say(event="note", text=f"voice separation skipped: {type(ex).__name__}: {ex}"[:300])
        vocals16 = load(spec["audio"], 16000, 1)[:, 0]
    if spec.get("vocals_out"):
        import soundfile as sf
        sf.write(spec["vocals_out"], vocals16, 16000)

    # 2. letters heard in the voice, every 20 ms
    say(event="aligning")
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    name = "facebook/wav2vec2-large-960h-lv60-self"
    proc = Wav2Vec2Processor.from_pretrained(name)
    asr = Wav2Vec2ForCTC.from_pretrained(name).to(device).eval()
    R = 16000
    chunk, ctx = 30 * R, 1 * R
    ems = []
    pos = 0
    x = vocals16 / (np.abs(vocals16).max() + 1e-6) * 0.9
    while pos < len(x):
        a0, a1 = max(0, pos - ctx), min(len(x), pos + chunk + ctx)
        inp = proc(x[a0:a1], sampling_rate=R, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            lg = torch.log_softmax(asr(inp).logits[0].float(), dim=-1).cpu()
        fps = lg.shape[0] / ((a1 - a0) / R)
        s = int(round((pos - a0) / R * fps))
        e = s + int(round(min(chunk, len(x) - pos) / R * fps))
        ems.append(lg[s:e])
        pos += chunk
    em = torch.cat(ems)                           # (frames, vocab)
    frame_sec = len(x) / R / em.shape[0]

    # 3. line the lyrics up with what was heard
    vocab = proc.tokenizer.get_vocab()
    blank = vocab.get("<pad>", 0)
    sep = vocab.get("|")
    import torchaudio.functional as F
    tokens = [[vocab[c] for c in w.upper() if c in vocab and c != "|"] for w in spec["words"]]
    sizes = spec.get("lines") or [len(spec["words"])]
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    text_of = [tuple(spec["words"][starts[i]:starts[i] + sizes[i]]) for i in range(len(sizes))]

    def align(seq):
        """Forced alignment of these lyric lines (original line numbers, repeats allowed) over
        the whole song. Returns {(position, word in line): {start, end, score[]}} in frames."""
        targets, owner = [], []
        for pos, li in enumerate(seq):
            for k in range(sizes[li]):
                t = tokens[starts[li] + k]
                if not t:
                    continue
                if targets:
                    targets.append(sep)
                    owner.append(None)
                targets += t
                owner += [(pos, k)] * len(t)
        if not targets or len(targets) >= em.shape[0]:
            return {}
        tg = torch.tensor([targets], dtype=torch.int32)
        ali, scores = F.forced_align(em[None], tg, blank=blank)
        spans = F.merge_tokens(ali[0], scores[0].exp(), blank=blank)
        got = {}
        for sp, o in zip(spans, owner):
            if o is None:
                continue
            w = got.setdefault(o, {"start": sp.start, "end": sp.end, "score": []})
            w["start"] = min(w["start"], sp.start)
            w["end"] = max(w["end"], sp.end)
            w["score"].append(float(sp.score))
        return got

    def span(got, pos, seq):
        ws = [got[(pos, k)] for k in range(sizes[seq[pos]]) if (pos, k) in got]
        if not ws:
            return None
        return (min(w["start"] for w in ws) * frame_sec, max(w["end"] for w in ws) * frame_sec,
                ws)

    # what the speech model hears in the voice: used to find singing that has no lyrics
    heard = []
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import ss_media as M
        heard = sorted(((w["s"] + w["e"]) / 2) for w in M.transcribe_raw(vocals16))
    except Exception as ex:
        say(event="note", text=f"listening check skipped: {type(ex).__name__}: {ex}"[:300])

    def unexplained(got, seq):
        """How many heard words have no lyric word on them: singing with no lyrics. Word by
        word, and no word may stretch past ~1.2 s, so lyrics spread thin over a section
        that's really sung twice don't count as covering it."""
        spans = sorted((w["start"] * frame_sec,
                        min(w["end"], w["start"] + int(1.2 / frame_sec)) * frame_sec)
                       for w in got.values())
        n, j = 0, 0
        for t in heard:
            while j < len(spans) and spans[j][1] + 0.45 < t:
                j += 1
            if not any(a - 0.45 <= t <= b + 0.45 for a, b in spans[j:j + 4]):
                n += 1
        return n

    def repeated_blocks(seq):
        """The longest runs of lyric lines that appear more than once (the chorus): the
        sections a singer might repeat more times than they're written."""
        keys = [text_of[li] for li in seq]
        n = len(seq)
        for size in range(min(14, n // 2), 1, -1):
            found = {}
            for i in range(n - size + 1):
                b = tuple(keys[i:i + size])
                if b in found or len(set(b)) < 2:
                    continue
                occ, j = [], 0
                while j <= n - size:
                    if tuple(keys[j:j + size]) == b:
                        occ.append(j)
                        j += size
                    else:
                        j += 1
                if len(occ) >= 2:
                    found[b] = occ
            if found:
                return [(size, occ) for occ in found.values()]
        return []

    seq = [i for i in range(len(sizes)) if sizes[i] > 0]
    added = []                     # [(position, how many lines)] of repeats that were added
    got = align(seq)
    def low_lines(got, seq):
        """Lines the aligner is unsure of (score under 0.5)."""
        n = 0
        for p in range(len(seq)):
            s = span(got, p, seq)
            sc = [x for w in s[2] for x in w["score"]] if s else []
            if not sc or sum(sc) / len(sc) < 0.5:
                n += 1
        return n

    if heard:
        base = unexplained(got, seq)
        base_cost = base + 3 * low_lines(got, seq)
        say(event="debug", text=f"as written: unheard {base}, cost {base_cost}")
        for _ in range(3):
            best = None
            for size, occ in repeated_blocks(seq):
                for j in occ:                    # try the repeat after each place it's sung
                    end = j + size
                    cand = seq[:end] + seq[j:end] + seq[end:]
                    g = align(cand)
                    if not g:
                        continue
                    u = unexplained(g, cand)
                    unsure = low_lines(g, cand)
                    cost = u + 3 * unsure        # a repeat in the wrong place leaves lines unsure
                    say(event="debug", text=f"repeat {size} lines after {end}: unheard {u}, "
                                            f"unsure lines {unsure}, cost {cost}")
                    if best is None or cost < best[5]:
                        best = (u, cand, g, end, size, cost)
            if not best or best[5] > base_cost - max(4, 0.2 * base_cost):
                break                            # adding a repeat doesn't explain the singing
            say(event="note", text=f"added a repeated section ({best[4]} lines): heard "
                                   f"{base} words without lyrics, now {best[0]}")
            base, seq, got, base_cost = best[0], best[1], best[2], best[5]
            added.append((best[3], best[4]))

    # lines that are written but not sung there get squeezed into a gap, where the aligner is
    # unsure of them (score far below the ~0.9+ of sung lines). Take out the least sure line,
    # line up the rest again, and repeat, so one mismatch can't throw off the song.
    dropped = []
    for _ in range(max(1, min(12, len(seq) // 3 + 1))):
        def line_score(p):
            s = span(got, p, seq)
            sc = [x for w in s[2] for x in w["score"]] if s else []
            return sum(sc) / len(sc) if sc else 0.0

        def squeezed(p):
            """Crammed into far less time than singing it takes (under 0.2 s a word). A line
            sung with different words still fills its own time, so it isn't squeezed."""
            s = span(got, p, seq)
            return not s or (s[1] - s[0]) / len(s[2]) < 0.2

        def repeat_nearby(p):
            """The same words as a line close by (a chorus or line written out once too
            often): both copies share the singing of one, so each is only half sure."""
            near = seq[max(0, p - 3):p] + seq[p + 1:p + 4]
            return any(text_of[q] == text_of[seq[p]] for q in near)
        doubtful = sorted((line_score(p), p) for p in range(len(seq))
                          if squeezed(p) or (repeat_nearby(p) and line_score(p) < 0.55))
        worst, wp = doubtful[0] if doubtful else (1.0, None)
        if worst >= 0.6 or len(seq) <= 2:
            break
        dropped.append(seq[wp])
        say(event="note", text=f"line {seq[wp] + 1} doesn't seem to be sung there; left out")
        seq = seq[:wp] + seq[wp + 1:]
        got = align(seq)

    out_lines = []
    for p, li in enumerate(seq):
        ws = []
        for k in range(sizes[li]):
            w = got.get((p, k))
            ws.append(None if w is None else {
                "start": round(w["start"] * frame_sec, 3), "end": round(w["end"] * frame_sec, 3),
                "score": round(sum(w["score"]) / len(w["score"]), 3)})
        out_lines.append({"line": li, "words": ws})
    Path(spec["out"]).write_text(json.dumps({
        "lines": out_lines, "dropped_lines": dropped,
        "added": [{"lines": n} for _, n in added], "heard_words": len(heard)}), encoding="utf8")
    say(event="done")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        say(event="fatal", text=f"{type(ex).__name__}: {ex}"[:500])
