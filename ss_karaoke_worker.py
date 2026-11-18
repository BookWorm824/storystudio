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
    heard, heard_words = [], []
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import ss_media as M
        heard_words = sorted(M.transcribe_raw(vocals16), key=lambda w: w["s"])
        heard = [(w["s"] + w["e"]) / 2 for w in heard_words]
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

    # singing that doesn't match the lyrics shown there, and isn't anywhere in the lyrics
    # (a verse that wasn't pasted): the lyrics around it get stretched over it, so the words
    # light up at the wrong time. Found by comparing what the speech model heard with the
    # lyric shown at that moment; then that stretch is kept free of lyrics and they're lined
    # up again, kept only if it agrees better with what was heard everywhere else.
    import difflib
    import re as _re

    def letters(s):
        return _re.sub(r"[^A-Z]", "", s.upper())

    lyric_letters = [letters(w) for w in spec["words"]]

    def alike(a, b):
        return bool(a and b) and (a == b or difflib.SequenceMatcher(None, a, b).ratio() >= 0.6)

    def heard_vs_shown(got, seq):
        """For each heard word: True = a lyric word like it is shown there, False = not,
        None = too short to judge."""
        shown = sorted((w["start"] * frame_sec, w["end"] * frame_sec,
                        lyric_letters[starts[seq[p]] + k]) for (p, k), w in got.items())
        out = []
        for hw, t in zip(heard_words, heard):
            h = letters(hw["t"])
            if len(h) < 3:
                out.append(None)
                continue
            near = [l for a, b, l in shown if a - 0.4 <= t <= b + 0.4]
            out.append(any(alike(h, l) for l in near))
        return out

    def in_lyrics(words):
        """How well these heard words match some stretch of the lyrics (0-1)."""
        H = [letters(w["t"]) for w in words if len(letters(w["t"])) >= 2]
        if not H:
            return 1.0
        best = 0.0
        L = [x for x in lyric_letters if x]
        for i in range(0, max(1, len(L) - len(H) + 1)):
            r = difflib.SequenceMatcher(None, H, L[i:i + len(H) + 3], autojunk=False).ratio()
            best = max(best, r)
        return best

    def unknown_spans(got, seq):
        """Runs of heard words that mostly don't match the lyrics shown with them, and don't
        match any part of the lyrics either: [(start, end, words)]."""
        flags = heard_vs_shown(got, seq)
        runs, cur = [], []
        for w, f in zip(heard_words, flags):
            if cur and w["s"] - cur[-1][0]["e"] > 2.5:
                runs.append(cur)
                cur = []
            if f is False or (cur and f is None):
                cur.append((w, f))
            elif f is True and cur:
                bad = sum(1 for _, x in cur if x is False)
                good = sum(1 for _, x in cur if x is True) + 1
                if good / (bad + good) > 0.35:
                    runs.append(cur)
                    cur = []
                else:
                    cur.append((w, f))
        if cur:
            runs.append(cur)
        out = []
        for r in runs:
            while r and r[-1][1] is not False:
                r = r[:-1]
            ws = [w for w, _ in r]
            if sum(1 for _, x in r if x is False) >= 4 and ws and ws[-1]["e"] - ws[0]["s"] >= 2.0 \
                    and in_lyrics(ws) < 0.5:
                out.append((ws[0]["s"], ws[-1]["e"], ws))
        return out

    unknown = []
    if heard_words:
        found = unknown_spans(got, seq)
        if found:
            masked = em.clone()
            for a, b, _ in found:
                f0, f1 = max(0, int((a - 0.3) / frame_sec)), int((b + 0.3) / frame_sec) + 1
                masked[f0:f1] = -1e4
                masked[f0:f1, blank] = 0.0
            inside = lambda t: any(a - 0.3 <= t <= b + 0.3 for a, b, _ in found)

            def wrong(g):
                return sum(1 for f, t in zip(heard_vs_shown(g, seq), heard)
                           if f is False and not inside(t))
            before = wrong(got)
            keep_em, em = em, masked
            g2 = align(seq)
            after = wrong(g2) if g2 else before + 1
            say(event="note", text=f"singing not in the lyrics at "
                                   f"{[(round(a, 1), round(b, 1)) for a, b, _ in found]}: "
                                   f"mismatched words elsewhere {before} -> {after}")
            if g2 and after <= before - 3:
                got = g2
                unknown = found
            em = keep_em

    def unsung(got):
        """Stretches of singing with no lyric word on them (an extra verse that wasn't pasted,
        a spoken bit): what was heard there, so the user can add it."""
        spans = sorted((w["start"] * frame_sec,
                        min(w["end"], w["start"] + int(1.2 / frame_sec)) * frame_sec)
                       for w in got.values())
        free = [w for w, t in zip(heard_words, heard)
                if not any(a - 0.45 <= t <= b + 0.45 for a, b in spans)]
        groups = []
        for w in free:
            if groups and w["s"] - groups[-1][-1]["e"] < 2.5:
                groups[-1].append(w)
            else:
                groups.append([w])
        spans_out = [{"start": round(a, 2), "end": round(b, 2), "kind": "new",
                      "heard": " ".join(x["t"] for x in ws)[:200]} for a, b, ws in unknown]
        for g in groups:
            if len(g) >= 4 and g[-1]["e"] - g[0]["s"] >= 2.0 and not any(
                    x["start"] - 1 <= g[0]["s"] <= x["end"] + 1 for x in spans_out):
                # words that ARE in the lyrics, sung again here: written once, sung twice
                spans_out.append({"start": round(g[0]["s"], 2), "end": round(g[-1]["e"], 2),
                                  "kind": "again" if in_lyrics(g) >= 0.5 else "new",
                                  "heard": " ".join(
                                      x["t"] for x in heard_words
                                      if g[0]["s"] - 0.05 <= x["s"] <= g[-1]["e"])[:200]})
        return sorted(spans_out, key=lambda x: x["start"])

    out_lines = []
    for p, li in enumerate(seq):
        ws = []
        for k in range(sizes[li]):
            w = got.get((p, k))
            if w is None:
                ws.append(None)
                continue
            a, b = w["start"] * frame_sec, w["end"] * frame_sec
            for u0, u1, _ in unknown:            # a word split around singing that isn't in the
                if a < u0 and b > u1:            # lyrics belongs after it, where it's sung
                    a = min(b - 0.12, u1 + 0.3)
            ws.append({"start": round(a, 3), "end": round(b, 3),
                       "score": round(sum(w["score"]) / len(w["score"]), 3)})
        out_lines.append({"line": li, "words": ws})
    Path(spec["out"]).write_text(json.dumps({
        "lines": out_lines, "dropped_lines": dropped, "unsung": unsung(got) if heard else [],
        "added": [{"lines": n} for _, n in added], "heard_words": len(heard)}), encoding="utf8")
    say(event="done")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        say(event="fatal", text=f"{type(ex).__name__}: {ex}"[:500])
