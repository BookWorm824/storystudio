"""Makes sound effects on this PC with MMAudio (video + text to synced audio).

Run by StoryStudio in its own process:  python ss_mmaudio_worker.py jobs.json
jobs.json: {"model": "large_44k_v2", "jobs": [{"video": path or null, "duration": sec,
            "prompt": str, "negative": str, "out": wav path, "seed": int}, ...]}
Prints one JSON line per event: {"event": "loading"|"done"|"error"|"finished", ...}
"""
import json
import os
import sys
import time
from pathlib import Path


def say(**kw):
    print(json.dumps(kw), flush=True)


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf8"))
    home = Path(spec.get("models_dir") or ".")
    home.mkdir(parents=True, exist_ok=True)
    os.chdir(home)                                   # MMAudio downloads into ./weights
    os.environ.setdefault("HF_HOME", str(home / "hf"))

    import torch
    import soundfile as sf
    from mmaudio.eval_utils import all_model_cfg, generate, load_video
    from mmaudio.model.flow_matching import FlowMatching
    from mmaudio.model.networks import get_my_mmaudio
    from mmaudio.model.utils.features_utils import FeaturesUtils

    # Pick the biggest model that fits in the graphics memory that is FREE right now
    # (Premiere and other apps share the card). Too little free: use the processor.
    ORDER = ["large_44k_v2", "medium_44k", "small_44k"]
    NEED_GB = {"large_44k_v2": 7.0, "medium_44k": 5.2, "small_44k": 3.8}
    device = spec.get("device") or "auto"
    want = spec.get("model") or "auto"
    free_gb = 0.0
    if device != "cpu" and torch.cuda.is_available():
        try:
            free_gb = torch.cuda.mem_get_info()[0] / 2 ** 30
        except Exception:
            free_gb = 0.0
    if want == "auto":
        fits = [m for m in ORDER if NEED_GB[m] <= free_gb]
        want = fits[0] if fits else "small_44k"
        if not fits:
            device = "cpu"
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    say(event="loading", model=want, device=device, free_gb=round(free_gb, 1))
    try:
        cfg = all_model_cfg[want]
        cfg.download_if_needed()
        net = get_my_mmaudio(cfg.model_name).to(device, dtype).eval()
        net.load_weights(torch.load(cfg.model_path, map_location=device, weights_only=True))
        feature_utils = FeaturesUtils(tod_vae_ckpt=cfg.vae_path,
                                      synchformer_ckpt=cfg.synchformer_ckpt,
                                      enable_conditions=True, mode=cfg.mode,
                                      bigvgan_vocoder_ckpt=cfg.bigvgan_16k_path,
                                      need_vae_encoder=False).to(device, dtype).eval()
    except Exception as ex:
        if device == "cuda" and _is_gpu_error(ex):
            say(event="gpu_full", model=want)       # the app retries smaller / on the processor
            sys.exit(3)
        say(event="fatal", text=f"The sound AI couldn't load: {type(ex).__name__}: {ex}"[:400])
        return
    say(event="ready", model=cfg.model_name, device=device)
    seq_cfg = cfg.seq_cfg
    fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=int(spec.get("steps", 25)))

    take_times = []
    for i, job in enumerate(spec["jobs"]):
        try:
            rng = torch.Generator(device=device)
            rng.manual_seed(int(job.get("seed", 42)))
            duration = float(job["duration"])
            clip_frames = sync_frames = None
            if job.get("video"):
                info = load_video(Path(job["video"]), duration)
                clip_frames = info.clip_frames.unsqueeze(0)
                sync_frames = info.sync_frames.unsqueeze(0)
                duration = info.duration_sec
            seq_cfg.duration = duration
            net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len,
                                   seq_cfg.sync_seq_len)
            # several takes (different seeds) when the app wants to pick the best one
            for k, path in enumerate(job.get("cands") or [job["out"]]):
                if device == "cuda" and _crawling(torch, take_times, duration):
                    # another app (Premiere) filled the card after we loaded, so Windows is
                    # swapping graphics memory and every take crawls: hand over to a
                    # smaller model instead
                    say(event="gpu_full", index=i, model=cfg.model_name, reason="slow")
                    sys.exit(3)
                t_take = time.time()
                rng.manual_seed(int(job.get("seed", 42)) + 1000 * k)
                with torch.inference_mode():
                    audio = generate(clip_frames, sync_frames, [job["prompt"]],
                                     negative_text=[job.get("negative", "")],
                                     feature_utils=feature_utils, net=net, fm=fm, rng=rng,
                                     cfg_strength=float(job.get("cfg", 4.5)))
                audio = audio.float().cpu()[0]
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                sf.write(path, audio.numpy().T, seq_cfg.sampling_rate, subtype="FLOAT")
                take_times.append((time.time() - t_take) / max(1.0, duration))
            say(event="done", index=i, out=job["out"], duration=duration)
        except Exception as ex:                     # one bad job never stops the rest...
            if device == "cuda" and _is_gpu_error(ex):
                # ...but after a graphics-card error the card can't be used again in this
                # process: stop, and the app restarts the rest with a smaller model
                say(event="gpu_full", index=i, model=cfg.model_name)
                sys.exit(3)
            say(event="error", index=i, text=f"{type(ex).__name__}: {ex}"[:400])
    say(event="finished")


def _crawling(torch, take_times, duration):
    """True when the last two takes were very slow AND the card is nearly full."""
    if len(take_times) < 2:
        return False
    recent = take_times[-2:]
    try:
        free = torch.cuda.mem_get_info()[0] / 2 ** 30
    except Exception:
        return False
    # on a free card a take needs well under 2 s per second of sound
    return min(recent) > 4.0 and free < 0.5


def _is_gpu_error(ex):
    t = f"{type(ex).__name__} {ex}".lower()
    return any(k in t for k in ("out of memory", "cuda", "acceleratorerror", "cublas", "cudnn"))


if __name__ == "__main__":
    main()
