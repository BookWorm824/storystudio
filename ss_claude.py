"""StoryStudio and Claude: plans the sound effects from the story, the timeline and pictures.

Uses the Claude Code CLI (`claude -p --json-schema ...`) so it runs on the user's Claude plan,
no API key. Pictures go in as one stream-json message, so no tools are needed.
"""
import base64
import json
import os
import shutil
import subprocess
from pathlib import Path

from ss_core import APP_DATA, NOWIN, StoryError, fmt_time, log

MODEL = os.environ.get("STORYSOUND_CLAUDE_MODEL", "opus")
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
    if has_claude:
        return "claude"
    if has_gpt:
        return "chatgpt"
    return None


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
    return ask_claude(prompt, schema, system, images, timeout)


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
           "--strict-mcp-config", "--model", MODEL, "--no-session-persistence"]
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
    if result.get("is_error") or "structured_output" not in result:
        _fail(str(result.get("result") or r.stderr or r.stdout))
    return result["structured_output"]


# ---------------------------------------------------------------------------
# the sound plan
# ---------------------------------------------------------------------------
KINDS = ["ambience", "footsteps", "action", "impact", "whoosh", "animal", "nature", "weather",
         "water", "vehicle", "household", "magic", "cartoon", "crowd", "music", "other"]
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
- FILL THE MOVIE. There should be no dead air: every moment has its place's background, and \
foley carries on through the narration too (kept soft under the voice), not only in the gaps. \
Wordless reactions (laughs, chuckles, sighs, gasps) sound best in the small gaps between lines.
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
            "movement and action, all the way through (soft under the narration).",
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


MUSIC_ON = """BACKGROUND MUSIC is wanted too. Where the timeline has NO music clip already, add kind "music" cues: one per scene or mood (usually 15 to 90 seconds, starting and ending at shot changes), instrumental only, gentle enough to sit under the narration. For each: sound_prompt = short instrumental description for an AI ("soft storybook piano and pizzicato strings, warm and curious, instrumental"); suno_prompt = a Suno style prompt with genre, mood, instruments, tempo and "instrumental", e.g. "gentle storybook orchestral, soft piano, pizzicato strings, light glockenspiel, warm and curious, 90 bpm, instrumental, no vocals"; adobe_prompt = the same idea in a sentence; search_words = 2-4 words; level = background. Never put music where the timeline already has music."""
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
            c["start"] = max(lo - 1, float(c["start"]))
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
              (" (this is background music, so fill in suno_prompt too)."
               if cue.get("kind") == "music" else " (suno_prompt: empty string)."))
    return ask(prompt, REDO_SCHEMA, system=SYSTEM, timeout=300)
