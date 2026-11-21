"""StoryStudio and Claude: plans the sound effects from the story, the timeline and pictures.

Uses the Claude Code CLI (`claude -p --json-schema ...`) so it runs on the user's Claude plan,
no API key. Pictures go in as one stream-json message, so no tools are needed.
"""
import base64
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from ss_core import APP_DATA, NOWIN, StoryError, fmt_time, log, log_error
import ss_craft as craft

DEFAULT_MODEL = "opus"
# Every Claude model the user can pick in Settings: (id, name, note). "" is StoryStudio's
# usual choice. A family name on its own ("opus", "fable"...) always means the newest of that
# family that the Claude program on this computer knows; the dated ones pin one version.
# Anything not listed can still be typed in ("Another model").
CLAUDE_MODELS = [
    ("", "Opus, always the newest", "StoryStudio's usual choice: strong and quick."),
    ("sonnet", "Sonnet, always the newest", "Faster and lighter on your plan."),
    ("haiku", "Haiku, always the newest", "Fastest and lightest. Fine for simple jobs."),
    ("claude-opus-5-5", "Opus 5.5", ""),
    ("claude-opus-5", "Opus 5", ""),
    ("claude-opus-4-8", "Opus 4.8", ""),
    ("claude-opus-4-7", "Opus 4.7", ""),
    ("claude-opus-4-6", "Opus 4.6", ""),
    ("claude-sonnet-5-5", "Sonnet 5.5", ""),
    ("claude-sonnet-5", "Sonnet 5", ""),
    ("claude-sonnet-4-6", "Sonnet 4.6", ""),
    ("claude-haiku-5-5", "Haiku 5.5", ""),
    ("claude-haiku-4-5", "Haiku 4.5", ""),
    ("fable", "Fable, always the newest", "Only if you choose it: slower, and uses much more "
                                          "of your plan."),
    ("claude-fable-5-2", "Fable 5.2", ""),
    ("claude-fable-5-1", "Fable 5.1", ""),
    ("claude-fable-5", "Fable 5", ""),
]
LAST_FILE = APP_DATA / "claude" / "last_model.txt"


def last_model():
    """The model that really answered StoryStudio's last question to Claude ("" if none yet)."""
    try:
        return LAST_FILE.read_text(encoding="utf8").strip()
    except OSError:
        return ""


def _answered_by(result):
    """Note which model Claude really used. The Claude program can answer with a different
    one than asked (when a plan is busy, say): that is logged, and Fable answering when it
    wasn't chosen is logged as a warning."""
    used = [m for m in (result.get("modelUsage") or {}) if "haiku" not in m.lower()] or \
        list(result.get("modelUsage") or {})
    if not used:
        return
    name = ", ".join(used)
    if name != last_model():
        log("claude answered with", name, "(asked for", model() + ")")
        try:
            LAST_FILE.parent.mkdir(parents=True, exist_ok=True)
            LAST_FILE.write_text(name, encoding="utf8")
        except OSError:
            pass
    if any("fable" in m.lower() for m in used) and "fable" not in model().lower():
        log("WARNING: Claude answered with Fable although", model(), "was asked for")


def chosen_model():
    """The Claude model picked in Settings ("" = StoryStudio's usual choice)."""
    from ss_core import load_settings
    return str(load_settings().get("claude_model") or "").strip()


def model():
    """The model name handed to the Claude program: what's picked in Settings, or Opus.
    Fable is only ever used when it is picked there by name."""
    return os.environ.get("STORYSOUND_CLAUDE_MODEL") or chosen_model() or DEFAULT_MODEL


def nice_model(mid):
    """'claude-opus-5-5' -> 'Opus 5.5' (for showing which model answered)."""
    known = next((n for i, n, _ in CLAUDE_MODELS if i == mid), "")
    if known:
        return known
    m = re.match(r"claude-([a-z]+)-(\d+)(?:-(\d+))?", mid or "")
    return f"{m.group(1).title()} {m.group(2)}{'.' + m.group(3) if m.group(3) and len(m.group(3)) < 3 else ''}" \
        if m else (mid or "")


def model_label(m=None):
    m = chosen_model() if m is None else m
    return next((name for i, name, _ in CLAUDE_MODELS if i == m), m)
LONG_ARGS = 24_000


def find_claude():
    exe = shutil.which("claude")
    if exe:
        return exe
    home = Path.home()
    for p in (home / ".local" / "bin" / "claude.exe", home / ".local" / "bin" / "claude",
              Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "claude" / "claude.exe"):
        if p.exists():
            return str(p)
    return None


def _env():
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)        # always use the Claude plan, never a paid key
    return env


def _fail(text):
    low = (text or "").lower()
    if "selected model" in low or ("model" in low and any(
            k in low for k in ("may not exist", "not have access", "not found", "invalid model",
                               "does not exist"))):
        log("claude model refused:", model(), (text or "")[-300:])
        raise StoryError(f"Claude couldn't use the model chosen in Settings "
                         f"({model_label()}). It may not be part of your plan. Open Settings "
                         "(the gear at the top) and pick another model.")
    if any(k in low for k in ("not logged in", "/login", "authenticat", "401")):
        raise StoryError("Claude isn't signed in. Open Claude, sign in, then try again.")
    if any(k in low for k in ("usage limit", "rate limit", "limit reached", "quota")):
        raise StoryError("You've reached your Claude usage limit for now. It resets after a "
                         "few hours.")
    log("claude failed:", text[-1500:] if text else "")
    raise StoryError("Claude couldn't answer this time. Please try again.")


def provider():
    """Which AI answers: "claude" or "chatgpt" (Settings: automatic / Claude / ChatGPT).
    Automatic uses Claude when it is on this computer, otherwise ChatGPT."""
    import ss_gpt as G
    from ss_core import load_settings
    want = load_settings().get("ai", "auto")
    has_claude, has_gpt = bool(find_claude()), bool(G.find())
    if want == "chatgpt" and has_gpt:
        return "chatgpt"
    if want == "claude" and has_claude:
        return "claude"
    if has_claude and not (has_gpt and time.time() < _CLAUDE_DOWN[0]):
        return "claude"
    if has_gpt:
        return "chatgpt"
    return None


# Claude can't answer just now (signed out, or the plan's limit is reached): with Automatic
# chosen and ChatGPT set up, ChatGPT answers instead until this time
_CLAUDE_DOWN = [0.0]


def ai_name():
    return {"claude": "Claude", "chatgpt": "ChatGPT"}.get(provider(), "Claude")


def ask(prompt, schema, system=None, images=None, timeout=1200):
    """One question to the AI (Claude or ChatGPT); returns the structured answer (a dict)."""
    p = provider()
    if p is None:
        raise StoryError("This part needs Claude or ChatGPT, and neither is set up on this "
                         "computer yet. Open Settings (the gear at the top) to set up ChatGPT, "
                         "or install Claude.")
    if p == "chatgpt":
        import ss_gpt as G
        return G.ask(prompt, schema, system=system, images=images, timeout=timeout)
    try:
        return ask_claude(prompt, schema, system, images, timeout)
    except StoryError as ex:
        import ss_gpt as G
        from ss_core import load_settings
        msg = str(ex)
        if load_settings().get("ai", "auto") == "auto" and G.find() and                 ("signed in" in msg or "usage limit" in msg):
            try:
                ready = G.status() == "ready"
            except Exception:
                ready = False
            if ready:
                log("Claude can't answer (" + msg[:60] + "); using ChatGPT for now")
                _CLAUDE_DOWN[0] = time.time() + 30 * 60
                return G.ask(prompt, schema, system=system, images=images, timeout=timeout)
        raise


def ask_claude(prompt, schema, system=None, images=None, timeout=1200):
    """One question to Claude; returns the structured answer (a dict)."""
    exe = find_claude()
    if not exe:
        raise StoryError("Claude isn't set up on this computer.")
    work = APP_DATA / "claude"
    work.mkdir(parents=True, exist_ok=True)
    content = []
    for label, path in images or []:
        content.append({"type": "text", "text": label})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.b64encode(Path(path).read_bytes()).decode()}})
    content.append({"type": "text", "text": prompt})
    msg = json.dumps({"type": "user", "message": {"role": "user", "content": content}})
    cmd = [exe, "-p", "--input-format", "stream-json", "--output-format", "stream-json",
           "--verbose", "--json-schema", json.dumps(schema), "--tools", "",
           "--strict-mcp-config", "--model", model(), "--no-session-persistence"]
    if system:
        cmd += ["--system-prompt", system]
    try:
        r = subprocess.run(cmd, input=msg + "\n", capture_output=True, text=True,
                           encoding="utf8", errors="replace", timeout=timeout, cwd=str(work),
                           env=_env(), **NOWIN)
    except subprocess.TimeoutExpired:
        raise StoryError("Claude took too long to answer. Please try again.")
    result = {}
    for line in (r.stdout or "").splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "result":
            result = ev
    _answered_by(result)
    if result.get("is_error") or "structured_output" not in result:
        _fail(str(result.get("result") or r.stderr or r.stdout))
    return result["structured_output"]


# ---------------------------------------------------------------------------
# the sound plan
# ---------------------------------------------------------------------------
KINDS = ["ambience", "footsteps", "action", "impact", "whoosh", "animal", "nature", "weather",
         "water", "vehicle", "household", "magic", "cartoon", "crowd", "music", "other",
         "voice"]            # voice = a character's own wordless sound (gasp, grunt, giggle)
LEVELS = ["background", "soft", "medium", "loud"]

CUE = {
    "type": "object",
    "properties": {
        "start": {"type": "number", "description": "seconds from the start of the movie"},
        "end": {"type": "number", "description": "seconds; ambience runs the whole scene"},
        "kind": {"type": "string", "enum": KINDS},
        "what": {"type": "string",
                 "description": "short plain description for the user, e.g. 'Front door slams'"},
        "sound_prompt": {"type": "string",
                         "description": "concrete words for an AI sound generator"},
        "search_words": {"type": "string", "description": "2-4 words for a sound library"},
        "adobe_prompt": {"type": "string",
                         "description": "a full sentence for Adobe's sound effect generator"},
        "suno_prompt": {"type": "string",
                        "description": "music cues only: a style prompt for Suno; empty "
                                       "string for every other kind"},
        "synced": {"type": "boolean",
                   "description": "true if it must land exactly on a visible moment"},
        "level": {"type": "string", "enum": LEVELS},
    },
    "required": ["start", "end", "kind", "what", "sound_prompt", "search_words",
                 "adobe_prompt", "suno_prompt", "synced", "level"],
    "additionalProperties": False,
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "movie_type": {"type": "string",
                       "enum": ["narrated story", "dialogue story", "music video", "other"]},
        "look": {"type": "string", "enum": ["cartoon", "realistic", "storybook"]},
        "notes": {"type": "string",
                  "description": "one or two plain sentences about the sound approach"},
        "cues": {"type": "array", "items": CUE},
    },
    "required": ["movie_type", "look", "notes", "cues"],
    "additionalProperties": False,
}

SYSTEM = """You are an experienced film sound designer. You add sound effects to short movies \
that a storyteller makes with AI video tools and edits in Adobe Premiere Pro (mostly gentle, \
family-friendly storybooks, Bible stories and kids' songs). The movie already has its \
narration/dialogue and music; you add everything else so it feels like a real movie.

How you work:
- AMBIENCE: every scene/location gets a background bed for its whole length, built from \
real sounds you would hear there (birdsong in a garden, crickets at night, wind in the rafters \
and a horse shifting in straw in a barn, a distant tractor on a farm, a busy market, gentle \
waves). NEVER ask for "room tone", "silence", "quiet air", "stillness" or "a quiet room": AI \
sound makers turn those into static hiss. If a place is truly silent, give it no background. \
Backgrounds are steady textures only (birdsong, a breeze in leaves, crickets, distant traffic, a \
crowd murmur, wind): NEVER a ticking clock, dripping, a dog barking, a lawnmower, a horn or any \
sound with a beat or a one-off event, because a background is built from short pieces and those \
sounds repeat and stutter. If the place needs one (a clock, a far-off bark), make it a separate \
spot effect at one moment. Start and end ambience at the exact shot-change times you are given when the place changes. \
Keep one ambience per place; do not stack two of the same.
- SPOT EFFECTS for things you can SEE make a sound: footsteps when someone walks or runs, \
doors, things picked up or put down, splashes, animals, vehicles, wings flapping, a fall or \
bump, cloth rustle on a big movement, magic sparkles, pops, tools, eating, a hug or pat.
- STORY SOUNDS: every sound the story or narration mentions or clearly implies must be there \
at the moment it happens, even if it is only heard, not seen: a phone ringing or buzzing, a \
knock, a car arriving, rain starting, a horse calling for her foal. Animal characters get \
their voices at the moments they react (calls, whinnies, nickers, bleats, barks), and people \
get wordless reactions where the story has them (a soft chuckle, a happy laugh, a gasp, a \
sigh).
- NO DEAD AIR, BUT EVERY SOUND EARNS ITS PLACE. Every moment has its place's background. On \
top of that, add an effect only where a viewer would expect to hear something and would \
recognise it: a door, footsteps, a vehicle, an animal, water, dishes, a pencil writing, a trunk \
opening, something put down or dropped, a hug or a pat that is the point of the shot. Do NOT add \
sounds for small movements that make no real sound: hands gesturing, a sleeve or sweater \
shifting, someone leaning, turning, swallowing or breathing, a photo being held. Those come out \
as meaningless noise and make the story worse. Effects carry on through the narration too (kept \
soft under the voice). Wordless reactions (a laugh, a chuckle, a gasp) only where the story or \
the picture clearly has one, in the small gaps between lines.
- Match the look: cartoon/animated pictures get lighter, playful, slightly exaggerated sounds \
(boing, swish, pop) but still natural, never silly over a serious moment; realistic pictures \
get natural sounds.
- Keep it kind and calm for children: no scary, violent or harsh sounds unless the story \
clearly needs a gentle version (a soft thud, distant thunder).
- NEVER make speech, voices saying words, singing or humming. Wordless sounds are fine and \
wanted (animal voices, a chuckle, a laugh, a gasp, a sigh, a baby's coo, a crowd murmur \
without words). Music only when the request below asks for it (kind "music").
- Music videos (the sound is a song most of the time, little or no narration): keep it \
sparse; low ambience and only clear big visible actions.
- Skip branded intro/outro clips (channel intros like "Storytime Saturday" or "Safe Harbor") \
and anything the timeline already has a sound effect clip for.
- Times: read them from the yellow timestamps on the stills (one still per second) and the \
exact shot list. "start" is the moment the sound begins: find the first still where the action \
starts (a hand reaching, a door starting to move) and use that time, not the still where it is \
already happening. \
synced=true for sounds that must land on a visible moment (a door shutting, a foot landing, a \
splash); the program fine-tunes those against the picture. Spot effects last as long as the \
sound really does (0.3 to 8 seconds); a long action (walking for 6 s) is one cue.
- sound_prompt: concrete, short English for an AI sound generator: source + action + \
material + place, e.g. "small wooden door creaks open slowly, quiet indoor room". No \
character names, no story words it would not understand.
- search_words: 2-4 plain words for a sound library search, e.g. "door creak wooden".
- adobe_prompt: one full sentence for Adobe Firefly's sound effect generator describing the \
sound, its intensity, how it starts and ends, and its length in seconds, e.g. "A small \
wooden door slowly creaks open in a quiet cottage, soft and squeaky, about 2 seconds."
- level: how loud it should sit in the mix: background (ambience), soft, medium, loud. Effects support the story and never take over: while the narrator is speaking choose soft or medium; save loud for big moments in gaps between lines.
- "what": a short plain description the storyteller will read in a list."""

BUSY = {
    "light": "Keep it light: ambience for each place plus only the key moments (about one spot "
             "effect every 15-20 seconds).",
    "normal": "Normal amount: ambience for each place plus the clearly visible actions (roughly "
              "one spot effect every 5-10 seconds where there is action).",
    "lots": "Make it rich, like a feature film: ambience, and foley for nearly every visible "
            "action that really makes a sound, all the way through (soft under the "
            "narration). Rich means more of the sounds that belong, never noise for its own "
            "sake.",
    "cartoon": craft.SOUND_PLAN,
}


def _clip_list(seq_info, lo, hi):
    out = []
    for s in seq_info["shots"]:
        if s["end"] > lo and s["start"] < hi:
            out.append(f"{fmt_time(s['start'])}-{fmt_time(s['end'])}  ({s['start']:.2f}s-"
                       f"{s['end']:.2f}s)  {s['name']}")
    return "\n".join(out) or "(none)"


def _audio_list(seq_info, lo, hi):
    out = []
    for a in seq_info["audio"]:
        if a["end"] > lo and a["start"] < hi:
            out.append(f"{fmt_time(a['start'])}-{fmt_time(a['end'])}  [{a['tag'] or 'audio'}] "
                       f"{a['name']}")
    return "\n".join(out) or "(none)"


def _speech(lines, lo, hi):
    out = [f"{fmt_time(l['s'])}-{fmt_time(l['e'])}: {l['text']}" for l in lines
           if l["e"] > lo and l["s"] < hi]
    return "\n".join(out) or "(no speech heard)"


MUSIC_ON = """BACKGROUND MUSIC is wanted too. Where the timeline has NO music clip already, add kind "music" cues: one per scene or mood (usually 15 to 90 seconds, starting and ending at shot changes), instrumental only, gentle enough to sit under the narration. For each: sound_prompt = short instrumental description for an AI ("soft storybook piano and pizzicato strings, warm and curious, instrumental"); suno_prompt = a Suno style prompt with genre, mood, instruments, tempo and "instrumental", e.g. "gentle storybook orchestral, soft piano, pizzicato strings, light marimba, warm and curious, 90 bpm, instrumental, no vocals, no bells"; adobe_prompt = the same idea in a sentence; search_words = 2-4 words; level = background. Never put music where the timeline already has music. Music must never sound like holiday music: no glockenspiel, celesta, bells, chimes, sleigh bells, jingles, choir or church organ (with warm strings they sound like a Christmas carol). Use instruments like pizzicato strings, clarinet, flute, bassoon, tuba, xylophone, marimba, ukulele, piano, woodblock and light percussion."""
MUSIC_OFF = ("No music: the storyteller makes the music separately. Do not plan any \"music\" "
             "cues. Leave suno_prompt as an empty string.")


def plan_sounds(story, seq_info, lines, sheets, busy="normal", progress=None,
                per_request=10, music=False):
    """sheets: [(t_from, t_to, path)]. Returns the merged plan dict."""
    groups = [sheets[i:i + per_request] for i in range(0, len(sheets), per_request)] or [[]]
    plan = {"movie_type": "", "look": "", "notes": "", "cues": []}
    story_text = (story or "").strip()[:30000] or "(no story file given; use the narration)"
    for gi, grp in enumerate(groups):
        lo = grp[0][0] if grp else 0.0
        hi = (groups[gi + 1][0][0] if gi + 1 < len(groups) else seq_info["duration"]) \
            if grp else seq_info["duration"]
        if progress:
            progress(gi / len(groups), f"{ai_name()} is planning sounds for {fmt_time(lo)} to "
                                       f"{fmt_time(hi)}…")
        before = ""
        if plan["cues"]:
            last = [c for c in plan["cues"] if c["end"] > lo - 30][-12:]
            before = ("\n\nSounds already planned just before this part (continue ambience "
                      "naturally if the place hasn't changed; don't repeat these):\n" +
                      "\n".join(f"{fmt_time(c['start'])}-{fmt_time(c['end'])} {c['kind']}: "
                                f"{c['what']}" for c in last))
        prompt = f"""Movie: "{seq_info['name']}", {seq_info['duration']:.1f} seconds long.
This request covers {lo:.1f}s to {hi:.1f}s{f' (part {gi + 1} of {len(groups)})' if len(groups) > 1 else ''}.
{BUSY.get(busy, BUSY['normal'])}
{MUSIC_ON if music else MUSIC_OFF}

THE STORY (the storyteller's script):
{story_text}

SHOTS ON THE TIMELINE in this part (exact times; file names sometimes describe the shot):
{_clip_list(seq_info, lo, hi)}

AUDIO ALREADY ON THE TIMELINE in this part:
{_audio_list(seq_info, lo, hi)}

WHAT IS SAID / SUNG (speech recognition, may mishear names):
{_speech(lines, lo, hi)}{before}

The pictures are stills from this part with their times in yellow. Plan every sound effect \
for {lo:.1f}s to {hi:.1f}s only. Give times in seconds from the start of the movie."""
        images = [(f"Stills {fmt_time(a)} to {fmt_time(b)}:", p) for a, b, p in grp]
        res = ask(prompt, PLAN_SCHEMA, system=SYSTEM, images=images)
        if not plan["movie_type"]:
            plan.update(movie_type=res.get("movie_type", ""), look=res.get("look", ""),
                        notes=res.get("notes", ""))
        for c in res.get("cues", []):
            if c.get("kind") == "music" and not music:
                continue
            c["start"] = min(hi, max(lo - 1, float(c["start"])))
            c["end"] = min(hi + 30, max(c["start"] + 0.2, float(c["end"])))
            plan["cues"].append(c)
    if progress:
        progress(1.0, f"{ai_name()} finished the sound plan.")
    return plan


MISSING_SCHEMA = {
    "type": "object",
    "properties": {"cues": {"type": "array", "items": CUE}},
    "required": ["cues"],
    "additionalProperties": False,
}


def check_missing(story, seq_info, lines, cues, music=False, progress=None):
    """A second read of the story and the timed narration against the plan: every sound the
    story mentions or implies (a phone ringing, a horse calling for her foal, a laugh, a
    knock) that isn't planned yet gets added at the moment it happens."""
    if progress:
        progress(0.0, f"{ai_name()} is checking the story for sounds that are still missing…")
    planned = "\n".join(f"{fmt_time(c['start'])}-{fmt_time(c['end'])} {c['kind']}: {c['what']}"
                        for c in sorted(cues, key=lambda c: c["start"])) or "(nothing yet)"
    dur = seq_info["duration"]
    prompt = f"""Movie: "{seq_info['name']}", {dur:.1f} seconds long.

THE STORY (the storyteller's script):
{(story or '').strip()[:30000] or '(no script; use the narration)'}

WHAT IS SAID, WITH TIMES (speech recognition, may mishear names):
{_speech(lines, 0, dur)}

SHOTS (exact times; file names sometimes describe the shot):
{_clip_list(seq_info, 0, dur)}

SOUNDS ALREADY PLANNED:
{planned}

Go through the story and the narration moment by moment. List ONLY sounds that are missing \
from the plan: anything the story or narration mentions or clearly implies happening (a phone \
ringing or buzzing, a knock, a car or trailer arriving, rain, a horse calling for her foal, a \
foal answering, a man chuckling, a woman laughing, hooves scrambling, a fence creaking). Put \
each at the moment it happens, using the speech times. Don't repeat anything already planned \
and don't add filler. {MUSIC_OFF if not music else ''} Give times in seconds."""
    res = ask(prompt, MISSING_SCHEMA, system=SYSTEM, timeout=900)
    out = []
    for c in res.get("cues", []):
        if c.get("kind") == "music" and not music:
            continue
        c["start"] = max(0.0, min(dur, float(c["start"])))
        c["end"] = min(dur, max(c["start"] + 0.3, float(c["end"])))
        c["added_by_check"] = True
        out.append(c)
    if progress:
        progress(1.0, f"Found {len(out)} more sounds the story needs.")
    return out


LOOK_SOUND = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "seen": {"type": "string", "enum": ["on screen", "off screen", "not there"],
                 "description": "on screen = you can see what makes it; off screen = heard "
                                "but not meant to be seen; not there = the pictures show it "
                                "doesn't happen in this shot"},
        "start": {"type": "number", "description": "exact second the sound begins"},
        "end": {"type": "number", "description": "second the sound stops"},
        "what": {"type": "string"},
        "sound_prompt": {"type": "string"},
        "search_words": {"type": "string"},
        "adobe_prompt": {"type": "string"},
    },
    "required": ["id", "seen", "start", "end", "what", "sound_prompt", "search_words",
                 "adobe_prompt"],
    "additionalProperties": False,
}
LOOK_SCHEMA = {
    "type": "object",
    "properties": {"sounds": {"type": "array", "items": LOOK_SOUND},
                   "missed": {"type": "array", "items": CUE}},
    "required": ["sounds", "missed"],
    "additionalProperties": False,
}


def _look_once(grp, name, story, look, busy):
    parts, images = [], []
    for k, s in enumerate(grp):
        step = (s["end"] - s["start"]) / max(1, s["n"])
        planned = "\n".join(
            f"  id {c['id']}: {c['start']:.2f}-{c['end']:.2f}s {c['kind']}: {c['what']} "
            f"(sound: {c['sound_prompt']})" for c in s["sounds"]) or "  (no sounds planned yet)"
        fname = f" (file: {s['name']})" if s.get("name") else ""
        parts.append(f"SHOT {k + 1}: {s['start']:.2f}s to {s['end']:.2f}s, a still every "
                     f"{step:.2f}s{fname}\n"
                     f"Planned sounds in this shot:\n{planned}")
        for j, (a, b, p) in enumerate(s["sheets"]):
            images.append((f"Shot {k + 1}, stills {a:.2f}s to {b:.2f}s:", p))
    prompt = f"""Movie: "{name}" ({look or 'story'} look). This is a CLOSE LOOK at {len(grp)} \
shots. Each shot's stills are close together, and the exact time in seconds is printed in \
yellow on every still.

THE STORY (for context):
{(story or '').strip()[:8000] or '(no script)'}

{chr(10).join(parts)}

For EVERY planned sound above, answer with its id:
- seen: "on screen" if you can see what makes it, "off screen" if it is heard but not meant to \
be seen (birds, a dog in the next room, distant thunder, a wordless reaction from someone out \
of frame), "not there" only if these pictures clearly show it doesn't happen in ANY of these \
shots. If it really happens in another shot of this request (the plan was a little early or \
late), say "on screen" and give the time where it happens.
- start: the exact second the sound begins: the still where the foot lands, the door starts \
to move, the lid hits, the hand touches the paper. Read it from the times on the stills; you \
may choose a time between two stills. For a sound that goes on (rustling, writing, walking) \
give when it starts.
- end: when it stops (a short hit lasts about 0.3 to 1 second).
- what, sound_prompt, search_words, adobe_prompt: repeat them unchanged, or correct them if \
the pictures show something different (what is really being handled, what it is made of, how \
hard, how fast). Same style as the originals: "what" is a short label for the storyteller, \
with no notes or comments in brackets.

Then "missed": sounds of things you can SEE making a noise in these shots that are not \
planned (a cup set down, a door closing, a page turning, a pencil scratching on paper, \
footsteps of someone walking in frame). Only clear visible actions with times read from the \
stills; no ambience, no music, no voices with words, no filler. {BUSY.get(busy, BUSY['normal'])}"""
    return ask(prompt, LOOK_SCHEMA, system=SYSTEM, images=images, timeout=900)


def close_look(shots, name, story, look, busy="normal", progress=None, cancel=None,
               per_request=6, workers=3):
    """A second, closer look at each shot (stills a few tenths of a second apart): every
    planned sound is timed to the exact moment, corrected or found not to be there, and
    sounds of visible actions that were missed are added. shots: [{"start", "end", "name",
    "n", "sheets": [(a, b, path)], "sounds": [cue with "id"]}].
    Returns {"sounds": {id: answer}, "missed": [cue]}."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    groups = [shots[i:i + per_request] for i in range(0, len(shots), per_request)]
    out = {"sounds": {}, "missed": []}
    lock, done = threading.Lock(), [0]

    def one(grp):
        if cancel and cancel():
            return
        try:
            res = _look_once(grp, name, story, look, busy)
        except StoryError as e:
            log("close look skipped for", len(grp), "shots:", e)
            res = {}
        except Exception:
            log_error("close look")
            res = {}
        lo, hi = grp[0]["start"], grp[-1]["end"]
        ids = {c["id"] for s in grp for c in s["sounds"]}
        with lock:
            for a in res.get("sounds", []):
                if a.get("id") in ids:
                    out["sounds"][a["id"]] = dict(a, lo=lo, hi=hi)   # what it could see
            for c in res.get("missed", []):
                try:
                    t = float(c["start"])
                except (TypeError, ValueError, KeyError):
                    continue
                if c.get("kind") in ("music", "ambience") or not lo <= t < hi:
                    continue
                c["start"], c["end"] = t, max(t + 0.2, float(c.get("end") or t + 0.5))
                c["added_by_look"] = True
                out["missed"].append(c)
            done[0] += 1
            if progress:
                progress(done[0] / len(groups), f"{ai_name()} is taking a close look at "
                                                f"each shot ({done[0]} of {len(groups)})…")
    if progress:
        progress(0.0, f"{ai_name()} is taking a close look at each shot…")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, groups))
    return out


# ---------------------------------------------------------------------------
# a cartoon's character voices: wordless sounds that show feeling, thought and intention
# ---------------------------------------------------------------------------
CAST_SCHEMA = {
    "type": "object",
    "properties": {"voices": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "who": {"type": "string", "description": "the character's name, or a plain name "
                    "like 'the squirrel' when the story gives none"},
            "voice": {"type": "string", "description": "the kind of wordless noises this "
                      "character makes, as the opening words of a sound-generator prompt, "
                      "about 8-14 words, e.g. 'small cute cartoon squirrel, high-pitched "
                      "squeaky animal chitters and squeaks, never words'. An animal makes "
                      "animal noises; it must never sound like a person talking"},
        },
        "required": ["who", "voice"], "additionalProperties": False}}},
    "required": ["voices"], "additionalProperties": False,
}
VOICE_SOUND = {
    "type": "object",
    "properties": {
        "start": {"type": "number", "description": "exact second the voice sound begins"},
        "end": {"type": "number", "description": "second it stops"},
        "who": {"type": "string", "description": "which character makes it"},
        "feeling": {"type": "string", "description": "what the character feels, thinks or "
                    "means to do at this moment, in a few plain words"},
        "what": {"type": "string", "description": "short label for the storyteller: who makes "
                 "what sound and why, e.g. 'Chester gasps: he can't believe his luck'"},
        "sound_prompt": {"type": "string", "description": "for an AI sound generator: the "
                         "character's sound words first, exactly as given, then precisely "
                         "what noise is heard and the feeling in it. Describe the noise; "
                         "never a syllable or a word to be said, nothing in quotes"},
        "adobe_prompt": {"type": "string", "description": "the same as one full sentence"},
        "level": {"type": "string", "enum": LEVELS},
        "instead_of": {"type": "string", "description": "the id of an already planned sound "
                       "that is nothing but this same voice sound (so it isn't made twice); "
                       "otherwise an empty string"},
    },
    "required": ["start", "end", "who", "feeling", "what", "sound_prompt", "adobe_prompt",
                 "level", "instead_of"],
    "additionalProperties": False,
}
VOICES_SCHEMA = {"type": "object", "properties": {"sounds": {"type": "array",
                                                               "items": VOICE_SOUND}},
                 "required": ["sounds"], "additionalProperties": False}


def cast_voices(name, story, images):
    """Who is in the cartoon and how each one's wordless voice sounds: [{"who", "voice"}]."""
    prompt = (f"Cartoon: \"{name}\". Nobody speaks in it. These stills come from shots across "
              "the film.\n\nTHE SCRIPT (for context):\n"
              f"{(story or '').strip()[:6000] or '(no script)'}\n\n"
              "List each character who is on screen often enough to need his own sounds "
              "(usually one or two), and describe the wordless noises that character makes. "
              "Nobody in this cartoon ever speaks: an animal makes cute animal noises "
              "(squeaks, chitters, sniffs, grunts), never words and never a person's voice. "
              "They must fit the character's kind, size and nature, be cute and funny rather "
              "than real or scary, and be something an AI sound generator can make the same "
              "way every time.")
    res = ask(prompt, CAST_SCHEMA, system=SYSTEM, images=images, timeout=300)
    return [v for v in res.get("voices", []) if v.get("who") and v.get("voice")][:4]


def _voices_once(grp, name, story, cast):
    import ss_craft as craft
    parts, images = [], []
    for k, s in enumerate(grp):
        step = (s["end"] - s["start"]) / max(1, s["n"])
        planned = "\n".join(
            f"  id {c['id']}: {c['start']:.2f}-{c['end']:.2f}s {c['kind']}: {c['what']}"
            for c in s["sounds"]) or "  (none)"
        script = s.get("script") or {}
        said = ""
        if script.get("action") or script.get("title"):
            said = (f"What the storyteller's script says about this shot: "
                    f"{script.get('title', '')}. {script.get('action', '')}\n")
        fname = f" (file: {s['name']})" if s.get("name") else ""
        parts.append(f"SHOT {k + 1}: {s['start']:.2f}s to {s['end']:.2f}s, a still every "
                     f"{step:.2f}s{fname}\n{said}"
                     f"Sounds already planned in this shot:\n{planned}")
        for a, b, p in s["sheets"]:
            images.append((f"Shot {k + 1}, stills {a:.2f}s to {b:.2f}s:", p))
    who = "\n".join(f"- {v['who']}: {v['voice']}" for v in cast) or \
        "- (decide fitting wordless cartoon animal noises and use the same words every time)"
    prompt = f"""Cartoon: "{name}". Nobody speaks in it. Plan the CHARACTERS' OWN WORDLESS \
SOUNDS for these {len(grp)} shots. Each shot's stills are close together, and the exact time \
in seconds is printed in yellow on every still.

EACH CHARACTER'S SOUND (begin every sound_prompt with these words, exactly):
{who}

THE SCRIPT (for context):
{(story or '').strip()[:6000] or '(no script)'}

{chr(10).join(parts)}

{craft.VOICES}

The sounds already planned are made by a sound maker that watches the picture: it is good at \
objects, footsteps and bumps and poor at voices. So plan every voice sound here yourself, \
even where a planned sound mentions a gasp or a grunt. When a planned sound is nothing but \
that same voice sound, put its id in "instead_of" so it isn't made twice; a planned sound \
that also has an object in it (a squeak of stretching rubber with a grunt) stays as it is: \
leave "instead_of" empty.
Read every time from the stills. "what" is a short label for the storyteller with no notes \
in brackets. Level: "medium" for most, "loud" for a yelp or a triumphant squeal, "soft" for \
a sigh or a hum."""
    return ask(prompt, VOICES_SCHEMA, system=SYSTEM, images=images, timeout=900)


def plan_voices(shots, name, story, cast, progress=None, cancel=None, per_request=6,
                workers=3):
    """The characters' wordless voice sounds for every shot. shots: [{"start", "end", "name",
    "n", "sheets", "sounds": [planned, with "id"], "script": {...}}]. Returns the sounds, each
    with the times the request could see as "lo" and "hi"."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    groups = [shots[i:i + per_request] for i in range(0, len(shots), per_request)]
    out, lock, done = [], threading.Lock(), [0]

    def one(grp):
        if cancel and cancel():
            return
        try:
            res = _voices_once(grp, name, story, cast)
        except StoryError as e:
            log("voices skipped for", len(grp), "shots:", e)
            res = {}
        except Exception:
            log_error("character voices")
            res = {}
        lo, hi = grp[0]["start"], grp[-1]["end"]
        with lock:
            for v in res.get("sounds", []):
                try:
                    t = float(v["start"])
                except (TypeError, ValueError, KeyError):
                    continue
                if lo <= t < hi and (v.get("sound_prompt") or "").strip():
                    out.append(dict(v, lo=lo, hi=hi))
            done[0] += 1
            if progress:
                progress(done[0] / len(groups), f"{ai_name()} is working out the characters' "
                                                f"own sounds ({done[0]} of {len(groups)})…")
    if progress:
        progress(0.0, f"{ai_name()} is working out the characters' own sounds…")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, groups))
    return sorted(out, key=lambda v: float(v["start"]))


REDO_SCHEMA = {
    "type": "object",
    "properties": {"sound_prompt": {"type": "string"}, "search_words": {"type": "string"},
                   "adobe_prompt": {"type": "string"}, "suno_prompt": {"type": "string"},
                   "what": {"type": "string"}},
    "required": ["sound_prompt", "search_words", "adobe_prompt", "suno_prompt", "what"],
    "additionalProperties": False,
}


def describe(user_words, cue):
    """Turn the user's own words ("make it a bigger splash") into prompts for one sound."""
    prompt = (f"The storyteller wants to change this sound effect.\nCurrent: {cue['what']} — "
              f"prompt: {cue['sound_prompt']} (lasts {cue['end'] - cue['start']:.1f} s)\n"
              f"What they said: {user_words}\nWrite the new prompts" +
              (" (this is background music, so fill in suno_prompt too). Music must never sound like holiday music: no glockenspiel, celesta, bells, chimes, sleigh bells, jingles, choir or church organ (with warm strings they sound like a Christmas carol). Use instruments like pizzicato strings, clarinet, flute, bassoon, tuba, xylophone, marimba, ukulele, piano, woodblock and light percussion."
               if cue.get("kind") == "music" else " (suno_prompt: empty string)."))
    return ask(prompt, REDO_SCHEMA, system=SYSTEM, timeout=300)
