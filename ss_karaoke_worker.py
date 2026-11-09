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

    def align(word_ids):
        """Forced alignment of these words (in order) over the whole song."""
        targets, owner = [], []
        for wi in word_ids:
            if not tokens[wi]:
                continue
            if targets:
                targets.append(sep)
                owner.append(-1)
            targets += tokens[wi]
            owner += [wi] * len(tokens[wi])
        if not targets:
            return {}
        tg = torch.tensor([targets], dtype=torch.int32)
        ali, scores = F.forced_align(em[None], tg, blank=blank)
        spans = F.merge_tokens(ali[0], scores[0].exp(), blank=blank)
        got = {}
        for sp, o in zip(spans, owner):
            if o < 0:
                continue
            w = got.setdefault(o, {"start": sp.start, "end": sp.end, "score": []})
            w["start"] = min(w["start"], sp.start)
            w["end"] = max(w["end"], sp.end)
            w["score"].append(float(sp.score))
        return got

    # lines that are written but not sung there get squeezed into a gap, where the aligner is
    # unsure of them (score far below the ~0.9+ of sung lines). Take out the least sure line,
    # line up the rest again, and repeat, so one mismatch can't throw off the song.
    sizes = spec.get("lines") or [len(spec["words"])]
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    keep = [i for i in range(len(sizes)) if sizes[i] > 0]
    dropped = []
    words = {}
    for _ in range(max(1, min(12, len(keep) // 3 + 1))):
        ids = [starts[i] + k for i in keep for k in range(sizes[i])]
        words = align(ids)

        def line_score(i):
            sc = [s for k in range(sizes[i]) for s in words.get(starts[i] + k, {}).get("score", [])]
            return sum(sc) / len(sc) if sc else 1.0

        def squeezed(i):
            """Crammed into far less time than singing it takes (under 0.2 s a word). A line
            sung with different words still fills its own time, so it isn't squeezed."""
            ws = [words[starts[i] + k] for k in range(sizes[i]) if starts[i] + k in words]
            if not ws:
                return True
            span = (max(w["end"] for w in ws) - min(w["start"] for w in ws)) * frame_sec
            return span / len(ws) < 0.2
        def repeat_nearby(i):
            """The same words as a line close by (a chorus or line written out once too
            often): both copies share the singing of one, so each is only half sure."""
            me = spec["words"][starts[i]:starts[i] + sizes[i]]
            pos = keep.index(i)
            near = keep[max(0, pos - 3):pos] + keep[pos + 1:pos + 4]
            return any(spec["words"][starts[j]:starts[j] + sizes[j]] == me for j in near)
        doubtful = sorted((line_score(i), i) for i in keep
                          if squeezed(i) or (repeat_nearby(i) and line_score(i) < 0.55))
        worst, wi_line = doubtful[0] if doubtful else (1.0, None)
        if worst >= 0.6 or len(keep) <= 2:
            break
        keep.remove(wi_line)
        dropped.append(wi_line)
        say(event="note", text=f"line {wi_line + 1} doesn't seem to be sung there; left out")
    out = []
    for wi in range(len(spec["words"])):
        w = words.get(wi)
        if w is None:
            out.append(None)
        else:
            out.append({"start": round(w["start"] * frame_sec, 3),
                        "end": round(w["end"] * frame_sec, 3),
                        "score": round(sum(w["score"]) / len(w["score"]), 3)})
    Path(spec["out"]).write_text(json.dumps({"words": out, "dropped_lines": sorted(dropped)}),
                                 encoding="utf8")
    say(event="done")


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        say(event="fatal", text=f"{type(ex).__name__}: {ex}"[:500])
