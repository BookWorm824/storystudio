"""Sounds made by Fal.ai, on the storyteller's own Fal account (their key, their credit).

Two sound makers are used:
- Mirelo SFX watches a short clip of the movie and makes the sound effects to match what
  happens in it, so they land in step with the picture;
- ElevenLabs sound effects makes a sound from a description alone: the backgrounds (as
  smooth loops) and any effect that has to be made again from its words.

Fal runs requests in a queue: a request is sent, its status is asked for until it has
finished, then the result (a link to the sound) is fetched and downloaded. A clip of the
movie has to be on Fal's own file store before a sound maker can watch it.
"""
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ss_core import NOWIN, StoryError, ffmpeg, log

QUEUE = os.environ.get("STORYSTUDIO_FAL_QUEUE", "https://queue.fal.run/")
REST = os.environ.get("STORYSTUDIO_FAL_REST", "https://rest.fal.ai")
CDN = os.environ.get("STORYSTUDIO_FAL_CDN", "https://v3.fal.media")

SFX_VIDEO = "mirelo-ai/sfx-v1.5/video-to-audio"       # watches the clip: effects in step
SFX_TEXT = "fal-ai/elevenlabs/sound-effects/v2"       # from a description
# Fal's listed prices, dollars per second of sound (November 2026). Only for the estimate
# shown to the user; Fal's own billing page is what counts.
PRICE = {SFX_VIDEO: 0.01, SFX_TEXT: 0.002}
VIDEO_MAX, TEXT_MAX = 10.0, 22.0                      # longest sound per request, seconds
WORKERS = 4


class FalError(Exception):
    """One request that didn't work (the others carry on)."""


def _plain(status, body):
    """Problems that stop the whole run, in plain words; None for anything else."""
    low = (body or "").lower()
    if any(k in low for k in ("exhausted balance", "top up", "insufficient", "out of credit",
                              "user is locked", "payment required")) or status == 402:
        return ("Your Fal account is out of credit. Add credit at fal.ai (Dashboard, then "
                "Billing), then use “Make the missing” to finish the sounds.")
    if status in (401, 403):
        return ("Fal didn't accept your key. Open Settings (the gear at the top) and paste "
                "your Fal key again.")
    return None


class Fal:
    def __init__(self, key):
        self.key = (key or "").strip()
        self._cdn = None
        self._lock = threading.Lock()
        self._uploaded = {}

    # ---- plumbing -------------------------------------------------------------------
    def _send(self, method, url, body=None, headers=None, timeout=120, auth=True):
        h = {"User-Agent": "StoryStudio", "Accept": "application/json"}
        if auth:
            h["Authorization"] = f"Key {self.key}"
        data = None
        if isinstance(body, (bytes, bytearray)):
            data = bytes(body)
        elif body is not None:
            data = json.dumps(body).encode("utf8")
            h["Content-Type"] = "application/json"
        h.update(headers or {})
        req = urllib.request.Request(url, data=data, headers=h, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as ex:
            text = ex.read().decode("utf8", "replace")
            stop = _plain(ex.code, text)
            if stop:
                raise StoryError(stop)
            raise FalError(f"{ex.code}: {text[:300]}")
        except (urllib.error.URLError, TimeoutError, OSError) as ex:
            raise FalError(f"couldn't reach Fal: {ex}")

    def _json(self, method, url, body=None, **kw):
        status, raw = self._send(method, url, body, **kw)
        try:
            return json.loads(raw.decode("utf8") or "{}")
        except ValueError:
            raise FalError(f"Fal sent something unexpected ({status})")

    # ---- putting a clip where Fal can watch it ---------------------------------------
    def upload(self, path, content_type="video/mp4"):
        """A link to this file on Fal's file store (each file is sent once per run)."""
        path = str(path)
        with self._lock:
            if path in self._uploaded:
                return self._uploaded[path]
        data = Path(path).read_bytes()
        name = Path(path).name
        url = None
        try:                                        # Fal's own file store
            with self._lock:
                if not self._cdn or self._cdn[2] < time.time():
                    t = self._json("POST", f"{REST}/storage/auth/token?storage_type=fal-cdn-v3",
                                   {})
                    self._cdn = (t.get("token_type") or "Bearer", t["token"],
                                 time.time() + 240, (t.get("base_url") or CDN).rstrip("/"))
                kind, token, _, base = self._cdn
            r = self._json("POST", f"{base}/files/upload", data, auth=False, timeout=300,
                           headers={"Authorization": f"{kind} {token}",
                                    "Content-Type": content_type, "X-Fal-File-Name": name})
            url = r.get("access_url")
        except (FalError, KeyError) as ex:
            log("fal file store said no, trying the other way:", ex)
        if not url:                                 # the older way in
            r = self._json("POST", f"{REST}/storage/upload/initiate?storage_type=gcs",
                           {"file_name": name, "content_type": content_type})
            self._send("PUT", r["upload_url"], data, auth=False, timeout=300,
                       headers={"Content-Type": content_type})
            url = r["file_url"]
        with self._lock:
            self._uploaded[path] = url
        return url

    # ---- one request, start to finish ------------------------------------------------
    def run(self, endpoint, payload, cancel=None, timeout=900):
        sub = self._json("POST", QUEUE + endpoint, payload)
        status_url, result_url = sub.get("status_url"), sub.get("response_url")
        if not status_url or not result_url:
            raise FalError(f"Fal didn't take the request: {str(sub)[:200]}")
        t0, wait = time.time(), 0.8
        while True:
            if cancel and cancel():
                try:
                    self._send("PUT", sub.get("cancel_url") or (result_url + "/cancel"))
                except (FalError, StoryError):
                    pass
                raise StoryError("Stopped.")
            st = self._json("GET", status_url)
            if st.get("status") == "COMPLETED":
                if st.get("error"):
                    raise FalError(f"{st.get('error_type', '')} {st['error']}"[:300])
                break
            if time.time() - t0 > timeout:
                raise FalError("Fal took too long")
            time.sleep(wait)
            wait = min(3.0, wait * 1.3)
        return self._json("GET", result_url)

    def fetch(self, url, dest):
        """Download a result and save it as an ordinary WAV file."""
        _, raw = self._send("GET", url, auth=False, timeout=300)
        tmp = Path(str(dest) + ".part")
        tmp.write_bytes(raw)
        r = subprocess.run([ffmpeg(), "-y", "-v", "error", "-i", str(tmp), "-vn", "-ar", "48000",
                            "-ac", "2", "-c:a", "pcm_f32le", "-f", "wav", str(dest)],
                           capture_output=True, **NOWIN)
        try:
            tmp.unlink()
        except OSError:
            pass
        if r.returncode != 0 or not Path(dest).exists():
            raise FalError("the sound Fal sent couldn't be read")
        return str(dest)

    # ---- the two sound makers ---------------------------------------------------------
    def from_video(self, clip, prompt, duration, dests, cancel=None):
        """Sound effects made by watching `clip`. Saves one take per path in dests (at
        least two are made: that is the fewest Mirelo gives). Returns seconds billed."""
        duration = max(1.0, min(VIDEO_MAX, float(duration)))
        n = max(2, min(8, len(dests)))
        res = self.run(SFX_VIDEO, {"video_url": self.upload(clip), "text_prompt": prompt,
                                   "duration": round(duration, 2), "num_samples": n}, cancel)
        audio = res.get("audio") or []
        if isinstance(audio, dict):
            audio = [audio]
        if not audio:
            raise FalError("no sound came back")
        for a, dest in zip(audio, dests):
            self.fetch(a["url"], dest)
        return duration * n

    def from_text(self, prompt, duration, dest, loop=False, cancel=None):
        """A sound made from its description. Returns seconds billed."""
        duration = max(0.5, min(TEXT_MAX, float(duration)))
        res = self.run(SFX_TEXT, {"text": prompt, "duration_seconds": round(duration, 1),
                                  "prompt_influence": 0.4 if loop else 0.5,
                                  "loop": bool(loop)}, cancel)
        audio = res.get("audio") or {}
        if isinstance(audio, list):
            audio = audio[0] if audio else {}
        if not audio.get("url"):
            raise FalError("no sound came back")
        self.fetch(audio["url"], dest)
        return duration


def run_jobs(jobs, key, progress=None, cancel=None):
    """jobs: the same dicts the on-PC sound AI takes ({video, duration, prompt, cands, out,
    loop}). A job with a video is made by watching it; one without is made from its
    description. Returns ({index: error text}, {"seconds": {endpoint: seconds}, "cost": $})."""
    if not jobs:
        return {}, {"seconds": {}, "cost": 0.0}
    fal = Fal(key)
    for j in jobs:                                   # never reuse an old file by mistake
        for f in [j["out"]] + list(j.get("cands") or []):
            try:
                Path(f).unlink()
            except OSError:
                pass
    errors, done, used = {}, [0], {SFX_VIDEO: 0.0, SFX_TEXT: 0.0}
    lock = threading.Lock()
    stop = []                                        # a problem that ends the whole run

    def one(i):
        if stop or (cancel and cancel()):
            return
        j = jobs[i]
        try:
            if j.get("video"):
                if len(j["cands"]) < 2:              # Mirelo always makes at least two
                    j["cands"].append(str(Path(j["cands"][0]).with_name(
                        Path(j["cands"][0]).stem + "b.wav")))
                secs = fal.from_video(j["video"], j["prompt"], j["duration"], j["cands"],
                                      cancel)
                with lock:
                    used[SFX_VIDEO] += secs
            else:
                for dest in j["cands"]:
                    secs = fal.from_text(j["prompt"], j["duration"], dest,
                                         loop=bool(j.get("loop")), cancel=cancel)
                    with lock:
                        used[SFX_TEXT] += secs
        except StoryError as ex:
            with lock:
                stop.append(str(ex))
        except Exception as ex:
            log("fal sound failed:", j.get("prompt", "")[:60], type(ex).__name__, ex)
            with lock:
                errors[i] = str(ex)[:200]
        with lock:
            done[0] += 1
            if progress and not stop:
                progress(done[0] / len(jobs), f"Making sounds with Fal: {done[0]} of {len(jobs)}")

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(one, range(len(jobs))))
    if stop:
        raise StoryError(stop[0])
    if cancel and cancel():
        raise StoryError("Stopped.")
    for i, j in enumerate(jobs):
        if i not in errors and not Path(j["cands"][0]).exists():
            errors[i] = "no sound came out"
    cost = sum(PRICE[k] * v for k, v in used.items())
    log("fal made sounds:", {k.split("/")[1]: round(v) for k, v in used.items()},
        f"seconds, about ${cost:.2f}; failed: {len(errors)} of {len(jobs)}")
    return errors, {"seconds": used, "cost": cost}


def estimate(seconds_synced, seconds_text, samples=2):
    """Rough dollars for a run, from Fal's listed prices."""
    return PRICE[SFX_VIDEO] * seconds_synced * samples + PRICE[SFX_TEXT] * seconds_text
