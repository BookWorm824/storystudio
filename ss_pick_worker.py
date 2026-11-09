"""Picks the best of several takes of each sound, the way paid sound generators do.

Run by StoryStudio:  python ss_pick_worker.py pick.json
pick.json: {"models_dir": ..., "groups": [{"prompt": str, "cands": [wav, ...], "out": wav}]}
Scores each take with CLAP (an AI that rates how well a sound matches words), drops takes
that are nearly silent or distorted, and copies the winner to "out".
Prints JSON lines: {"event": "loading"|"picked"|"finished", ...}
"""
import json
import os
import shutil
import sys
from pathlib import Path


def say(**kw):
    print(json.dumps(kw), flush=True)


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf8"))
    home = Path(spec.get("models_dir") or ".")
    os.environ.setdefault("HF_HOME", str(home / "hf"))
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import ClapModel, ClapProcessor

    device = "cpu"
    try:
        if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 1.5 * 2 ** 30:
            device = "cuda"
    except Exception:
        pass
    say(event="loading", device=device)
    name = "laion/clap-htsat-unfused"
    model = ClapModel.from_pretrained(name).to(device).eval()
    proc = ClapProcessor.from_pretrained(name)
    SR = 48000

    def load(path):
        a, sr = sf.read(path, dtype="float32", always_2d=True)
        a = a.mean(axis=1)
        if sr != SR:
            n = int(len(a) * SR / sr)
            a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype("float32")
        return a

    def flatness(x):
        """0 = clear sounds (birds, crickets, voices), 1 = pure static/hiss."""
        x = x[: SR * 10]
        fr = np.fft.rfftfreq(4096, 1 / SR)
        band = (fr > 300) & (fr < 8000)
        vals = []
        for i in range(0, len(x) - 4096, 4096):
            sp = np.abs(np.fft.rfft(x[i:i + 4096] * np.hanning(4096))) ** 2 + 1e-12
            if sp.sum() < 1e-6:
                continue
            b = sp[band]
            vals.append(np.exp(np.mean(np.log(b))) / np.mean(b))
        return float(np.median(vals)) if vals else 0.0

    for gi, g in enumerate(spec["groups"]):
        best, best_score, scores, flats = None, -9.0, [], {}
        try:
            with torch.no_grad():
                t = proc(text=[g["prompt"]], return_tensors="pt", padding=True).to(device)
                for c in g["cands"]:
                    if not Path(c).exists():
                        scores.append(None)
                        continue
                    a = load(c)
                    peak = float(np.abs(a).max()) if len(a) else 0.0
                    if peak < 0.01:                       # basically silent
                        scores.append(-1.0)
                        continue
                    clip = float((np.abs(a) > 0.995).mean())
                    # score the loudest 10 s (CLAP listens to 10 s at a time)
                    if len(a) > SR * 10:
                        # loudest 10 s window, from a running sum (fast for long sounds)
                        cs = np.concatenate([[0.0], np.cumsum(a.astype(np.float64) ** 2)])
                        starts = np.arange(0, len(a) - SR * 10 + 1, SR // 2)
                        e = cs[starts + SR * 10] - cs[starts]
                        s0 = int(starts[int(np.argmax(e))])
                        a = a[s0: s0 + SR * 10]
                    x = proc(audio=[a], sampling_rate=SR, return_tensors="pt").to(device)
                    out = model(**t, **x)            # text + sound in one pass
                    ae = out.audio_embeds / out.audio_embeds.norm(dim=-1, keepdim=True)
                    te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
                    s = float((ae @ te.T).item()) - clip * 5.0
                    if g.get("bed"):
                        # backgrounds: mark down takes that sound like static ("hashy")
                        f = flatness(a)
                        flats[c] = f
                        s -= 1.5 * max(0.0, f - 0.2)
                    scores.append(round(s, 4))
                    if s > best_score:
                        best, best_score = c, s
            if best is None:
                best = next((c for c in g["cands"] if Path(c).exists()), None)
            if best:
                shutil.copyfile(best, g["out"])
            say(event="picked", index=gi, best=best, scores=scores,
                flat=flats.get(best) if best else None)
        except Exception as ex:
            fallback = next((c for c in g["cands"] if Path(c).exists()), None)
            if fallback:
                shutil.copyfile(fallback, g["out"])
            say(event="picked", index=gi, best=fallback, error=f"{type(ex).__name__}: {ex}"[:300])
    say(event="finished")


if __name__ == "__main__":
    main()
