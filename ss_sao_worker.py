"""Makes long background sounds (and rough music) with Stable Audio Open.

Run by StoryStudio:  python ss_sao_worker.py jobs.json
jobs.json: {"models_dir": ..., "token": hf key, "jobs": [{"prompt", "negative", "duration",
            "cands": [wav, ...], "seed"}]}
Prints JSON lines: {"event": "loading"|"ready"|"done"|"error"|"no_access"|"fatal"|"finished"}
"""
import json
import os
import sys
from pathlib import Path

MODEL = "stabilityai/stable-audio-open-1.0"


def say(**kw):
    print(json.dumps(kw), flush=True)


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf8"))
    home = Path(spec.get("models_dir") or ".")
    os.environ.setdefault("HF_HOME", str(home / "hf"))
    import soundfile as sf
    import torch
    from diffusers import StableAudioPipeline

    token = spec.get("token") or None
    free = 0.0
    if torch.cuda.is_available():
        try:
            free = torch.cuda.mem_get_info()[0] / 2 ** 30
        except Exception:
            free = 0.0
    device = "cuda" if free >= 2.0 else "cpu"
    say(event="loading", device=device, free_gb=round(free, 1))
    try:
        pipe = StableAudioPipeline.from_pretrained(
            MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            token=token)
    except Exception as ex:
        t = f"{type(ex).__name__} {ex}".lower()
        if any(k in t for k in ("401", "403", "gated", "access", "token", "restricted",
                                "authenticat")):
            say(event="no_access", text=str(ex)[:300])
        else:
            say(event="fatal", text=f"{type(ex).__name__}: {ex}"[:400])
        return
    if device == "cuda":
        if free >= 6.5:
            pipe = pipe.to("cuda")
        else:
            pipe.enable_model_cpu_offload()      # shares the card with Premiere: slower, fits
    sr = pipe.vae.sampling_rate
    say(event="ready", device=device)
    steps = int(spec.get("steps", 100))
    for i, job in enumerate(spec["jobs"]):
        try:
            n = len(job["cands"])
            g = torch.Generator("cpu").manual_seed(int(job.get("seed", 1)))
            out = pipe(job["prompt"], negative_prompt=job.get("negative") or "Low quality.",
                       num_inference_steps=steps,
                       audio_end_in_s=float(min(47.0, max(2.0, job["duration"]))),
                       num_waveforms_per_prompt=n, generator=g).audios
            for k, path in enumerate(job["cands"]):
                a = out[k].float().cpu().numpy().T          # (samples, channels)
                sf.write(path, a, sr, subtype="FLOAT")
            say(event="done", index=i)
        except Exception as ex:
            if device == "cuda" and "out of memory" in str(ex).lower():
                torch.cuda.empty_cache()
            say(event="error", index=i, text=f"{type(ex).__name__}: {ex}"[:400])
    say(event="finished")


if __name__ == "__main__":
    main()
