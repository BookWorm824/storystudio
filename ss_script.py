"""Script builder: a story script or song lyrics -> a shot list with, for every shot, a
picture prompt (ChatGPT images / Nano Banana) and a MiniMax Hailuo 2.3/02 prompt.

    plan = build(spec, progress, cancel)

spec: {kind: "story"|"music", title, text, song_seconds, clip (5|8), style_id or custom style
       {name, style, clean}, aspect ("16:9"...)}

How it works:
1. The AI reads the whole script or lyrics: the title, how long the story will take to tell
   (stories) or what the music video is about (songs), the cast with one fixed look each, the
   places, and the scenes with their length.
2. Shots: one per clip (8 s for stories; 5 or 8 s for songs), plus about 20% extra shots
   (close-ups, cutaways, reactions) so there is always something to cut to.
3. Each shot gets a picture description (subject, action, place, framing, light: the Nano Banana
   formula) and a motion line for MiniMax with exactly one bracket camera command.
4. StoryStudio puts the prompts together itself, so the art style and every character's look are
   word-for-word the same in every shot.
Rules for MiniMax prompts: research/MiniMax Hailuo prompt rules.md.
"""
import json
import math
import re
import time
from pathlib import Path

from ss_core import APP_DIR, APP_DATA, StoryError, documents_dir, fmt_time, log, safe_name

CAMERA = ["Static shot", "Push in", "Pull out", "Pan left", "Pan right", "Truck left",
          "Truck right", "Tilt up", "Tilt down", "Pedestal up", "Pedestal down", "Zoom in",
          "Zoom out", "Tracking shot", "Shake"]
ASPECTS = {"16:9": "16:9 landscape", "9:16": "9:16 vertical portrait", "1:1": "1:1 square",
           "4:3": "4:3 landscape", "3:4": "3:4 portrait", "4:5": "4:5 portrait",
           "21:9": "21:9 ultra-wide cinematic"}
WORDS_PER_SEC = 2.2                        # a warm storytelling pace (about 130 words a minute)
EXTRA = 0.2                                # extra shots on top of what the timing needs
CUSTOM_FILE = APP_DATA / "my_styles.json"


# ---------------------------------------------------------------------------
# art styles: the ones from the Master Drawing Style Prompts PDF, plus the user's own
# ---------------------------------------------------------------------------
def styles():
    try:
        base = json.loads((APP_DIR / "ss_styles.json").read_text(encoding="utf8"))
    except (OSError, ValueError):
        base = {"universal_clean": "", "styles": []}
    mine = []
    try:
        mine = json.loads(CUSTOM_FILE.read_text(encoding="utf8"))
    except (OSError, ValueError):
        pass
    for s in mine:
        s["group"] = "My styles"
        s["mine"] = True
    return {"universal_clean": base.get("universal_clean", ""), "styles": mine + base["styles"]}


def save_style(name, style, clean=""):
    name = (name or "").strip()
    style = (style or "").strip()
    if not name or not style:
        raise StoryError("Give your style a name and describe it.")
    try:
        mine = json.loads(CUSTOM_FILE.read_text(encoding="utf8"))
    except (OSError, ValueError):
        mine = []
    sid = "my-" + (re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or str(int(time.time())))
    mine = [s for s in mine if s["id"] != sid]
    mine.insert(0, {"id": sid, "name": name, "style": style, "clean": (clean or "").strip()})
    CUSTOM_FILE.write_text(json.dumps(mine, indent=1, ensure_ascii=False), encoding="utf8")
    return sid


def remember_short(sid, short):
    """Keep the short version of one of the user's own styles, so it's only written once."""
    try:
        mine = json.loads(CUSTOM_FILE.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return
    for s in mine:
        if s["id"] == sid:
            s["style_short"] = short
    CUSTOM_FILE.write_text(json.dumps(mine, indent=1, ensure_ascii=False), encoding="utf8")


def delete_style(sid):
    try:
        mine = json.loads(CUSTOM_FILE.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return
    CUSTOM_FILE.write_text(json.dumps([s for s in mine if s["id"] != sid], indent=1,
                                      ensure_ascii=False), encoding="utf8")


def find_style(sid):
    for s in styles()["styles"]:
        if s["id"] == sid:
            return s
    raise StoryError("Please choose an art style.")


def improved(style):
    """True when the improved wording is used (the default) and this style has one."""
    return style.get("wording", "improved") != "original" and bool(style.get("style_opt"))


def style_text(style):
    """Just the look (no render standard), in the chosen wording."""
    return (style["style_opt"] if improved(style) else style["style"]).strip()


def style_block(style, aspect):
    """The style and its render standard, word for word, with the picture's shape swapped in.
    Improved wording (default): tighter and positively worded; original: the PDF's text."""
    shape = ASPECTS.get(aspect, aspect)
    if improved(style):
        block = f"{style['style_opt'].strip()} {style.get('clean_opt', '').strip()}".strip()
    else:
        text = style["style"].strip()
        clean = (style.get("clean") or "").strip()
        if not clean and "16:9" not in text:          # the standard isn't in the text already
            clean = styles()["universal_clean"].strip()
        block = text + ((" Clean Render Standard: " + clean) if clean else "")
    block = re.sub(r"16:9 landscape composition", f"{shape} composition", block)
    if aspect != "16:9" and shape not in block:
        block += f" {shape} composition."
    return block


def short_style(style):
    """The style in about 15 words, to end every shot prompt (like "cinematic realism, 16:9")."""
    s = (style.get("style_short") or "").strip().rstrip(".")
    if s:
        return s
    out = []                     # own styles without one: the first few phrases of the description
    for p in (p.strip() for p in re.split(r"[,;.]", style_text(style))):
        if not p:
            continue
        if out and len(" ".join(out + [p]).split()) > 18:
            break
        out.append(p)
    return ", ".join(out) or style.get("name", "")


def style_tag(style, aspect):
    return f"{short_style(style)}, {ASPECTS.get(aspect, aspect)} composition, no text or watermarks."


def shorten_style(style):
    """Ask the AI for a short version of one of the user's own styles (kept with the style)."""
    import ss_claude as C
    res = C.ask(f"""Write a SHORT style tag to end every text-to-image prompt for this art style: \
10-18 words, comma-separated phrases, starting with the style's name in plain words, then the 3-5 \
look words that make it recognisable (medium, rendering, line, colour, light). No picture shape \
and no rules. Put it in "text".

STYLE "{style.get('name', '')}": {style_text(style)}""",
                {"type": "object", "properties": {"text": {"type": "string"}},
                 "required": ["text"], "additionalProperties": False}, timeout=300)
    return res["text"].strip().rstrip(".")


# ---------------------------------------------------------------------------
# step 1: read the script
# ---------------------------------------------------------------------------
CHAR = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "look": {"type": "string", "description": "one fixed visual description used word for "
                 "word in every picture: species/age, build, face, hair (colour, length, style),"
                 " eyes, skin/fur, outfit with colours and materials, shoes, accessories. "
                 "Exactly ONE individual. Never mention or compare to another character (give "
                 "size in plain words like 'about one metre tall', never 'as tall as the "
                 "girl'), because the look is also used for a picture of this character alone. "
                 "Don't mention the art style or medium (3D, 2D, watercolor...); that is added "
                 "separately"},
        "short": {"type": "string", "description": "the same look in 12-20 words: the key "
                  "features that identify them at a glance (age or species, hair, skin, main "
                  "outfit colours), used word for word in every shot prompt"},
        "voice": {"type": "string", "description": "short note on personality, for posing"},
    },
    "required": ["name", "look", "short", "voice"],
    "additionalProperties": False,
}
SCENE = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "place": {"type": "string", "description": "the setting, described the same way every "
                  "time it returns (materials, colours, time of day, light)"},
        "text": {"type": "string", "description": "the exact part of the script or lyrics this "
                 "scene covers, copied word for word"},
        "seconds": {"type": "number", "description": "how long this part takes to tell or sing"},
        "mood": {"type": "string"},
        "section": {"type": "string", "description": "songs: Intro, Verse 1, Chorus, Bridge, "
                    "Outro...; stories: Opening, the part of the story (Problem, Journey...), "
                    "Ending"},
        "concept": {"type": "string", "description": "2-3 sentences: what this scene shows "
                    "and why it matters to the story"},
    },
    "required": ["title", "place", "text", "seconds", "mood", "section", "concept"],
    "additionalProperties": False,
}
PLAN = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string", "description": "two sentences: what the video shows"},
        "seconds": {"type": "number", "description": "story: how long the narrated story will "
                    "take in seconds; song: the song length you were given"},
        "characters": {"type": "array", "items": CHAR},
        "scenes": {"type": "array", "items": SCENE},
        "visual_rules": {"type": "string", "description": "2-3 sentences that keep the whole "
                         "video looking like one world: the settings, colours, how the light "
                         "changes through the video, the overall feel"},
        "approach": {"type": "string", "description": "3-5 short sentences: the user's "
                     "direction turned into plain rules for every shot: what kind of video "
                     "this is, who is on screen (a main character, different people each time, "
                     "or no people), the mix of shots, and anything the user asked for or "
                     "against. Every shot will be written from these rules."},
    },
    "required": ["title", "summary", "seconds", "characters", "scenes", "visual_rules",
                 "approach"],
    "additionalProperties": False,
}
SYSTEM = """You are a storyboard artist and director for Safe Harbor Studio, a wholesome family \
YouTube channel that makes music videos and stories for children and families with AI. Pictures \
are made in ChatGPT and Nano Banana, then each picture is animated in MiniMax Hailuo 2.3 into one \
short clip. Plan shots that an image AI can draw well and a video AI can animate well: clear \
subjects, one clear action per shot, believable motion.

THE USER'S DIRECTION COMES FIRST. Follow it exactly and literally, even when it differs from \
your own idea of a good video. Words like "B-roll", "no main character", "visualizer" or \
"montage" mean there is NO recurring character: different people (or none) in different shots.

WHAT THIS STUDIO WANTS (from the owner's own corrections on past scripts):
- Variety. Mix wide shots, medium action shots, close-ups and details; never more than two \
close-ups or head shots in a row. Don't let one subject (one animal, object, place or activity) \
take over unless the song is about it. Never repeat a scene or shot idea.
- Warm, human and wholesome: inviting light, friendly faces, people doing things together and \
interacting. Never cold, distant, gloomy, empty or "expensive"-looking; no lifeless shots of \
objects alone or people just standing and staring.
- Match the words: each shot shows what its line is about, so a viewer can tell which line is \
being sung or read. Use all the words, in order. A sad or frightening line for young children is \
shown gently and hopefully, not literally.
- Choruses are the big moments: brighter, wider, more movement, and their look returns and grows.

STUDIO RULES (always, in every prompt you write):
- Never show religious symbols of any kind (crosses, halos, angels' wings as symbols, church \
icons, praying-hands icons, holy books with symbols, stained-glass religious scenes), and no \
churches or church buildings.
- Jesus is never shown as a person or a figure. When the story is about Jesus or God, show only \
a warm, gentle light (a soft glow, rays of light, light filling the room) and the people \
reacting to it; if the light needs a shape, it is the plain shape of a man, without a robe or \
long hair.
- No holiday imagery: nothing for Christmas, Halloween, Easter or St. Patrick's Day (no Christmas \
trees, Santa, stockings, pumpkins or jack-o'-lanterns, witches, ghosts, Easter bunnies or eggs, \
shamrocks, leprechauns). Choose neutral settings and seasons instead.
- No alcohol. Couples are always clearly a married man and wife. No musicians, instruments, \
microphones or stage performances unless the direction asks for them. No "good works" or \
"get saved" messages. Nothing scary or intense for young children.
- Bible-era scenes: historically accurate everyday clothing of the time, not storybook robes. A \
Bible is a plain King James Bible, never a picture book. A cross, if the user asks for one, is \
only a T-shaped (Tau) cross, in shadow or backlit, shown tastefully.
- A family is all one nationality."""
RULES_LINE = ("Absolutely no religious symbols or churches; Jesus or God only ever as a warm "
              "gentle light, never a person; no Christmas, Halloween, Easter or St. Patrick's "
              "Day imagery; no alcohol; couples are a married man and wife.")

# who is on screen (the "Who's in it" choice)
ON_MAIN = ("WHO IS ON SCREEN: main character(s). The video follows the cast, but not every shot "
           "needs them: mix in places, nature and details around them.")
ON_NONE = ("WHO IS ON SCREEN: NO MAIN CHARACTER (B-roll style). There are no recurring "
           "characters, so the characters list is EMPTY. Show a variety of different people of "
           "all ages (children, parents, grandparents, married couples, friends): a different "
           "person or group in each shot with people, each described inside that shot. Mix them "
           "about half and half with nature, places, animals and close details. The people do "
           "warm, real things together that fit the words.")
ON_NOPEOPLE = ("WHO IS ON SCREEN: NO PEOPLE. Only nature, animals, places and objects: no "
               "people or human figures anywhere, not even far away. The characters list is "
               "EMPTY unless the direction asks for a recurring animal character.")


def on_screen(spec):
    """The who-is-on-screen rule for this video."""
    on = spec.get("onscreen") or "auto"
    if on == "main":
        return ON_MAIN
    if on == "none":
        return ON_NONE
    if on == "nopeople":
        return ON_NOPEOPLE
    return ("WHO IS ON SCREEN: decide from the direction. If it says B-roll, no main character, "
            "visualizer, montage or anything like it, follow this rule: " + ON_NONE + " If it "
            "says no people (only nature or animals), follow this rule: " + ON_NOPEOPLE +
            " Otherwise: " + ON_MAIN)


def read_script(spec, progress=None):
    import ss_claude as C
    text = spec["text"].strip()
    words = len(text.split())
    direction = (spec.get("direction") or "").strip()
    nobody = spec.get("onscreen") in ("none", "nopeople")
    if spec["kind"] == "story":
        task = f"""This is a STORY script (about {words} words). It will be read aloud as \
narration over the pictures. Work out how long the finished narrated story will take: a warm, \
unhurried storytelling pace is about {WORDS_PER_SEC * 60:.0f} words a minute, plus short pauses \
between scenes and moments where the picture needs to breathe (an opening, big moments, the \
ending). Give that total in "seconds".
{f"THE USER'S NOTES FOR THE PICTURES (follow them exactly): {direction}" if direction else ""}"""
    else:
        task = f"""These are SONG LYRICS for a music video. The song is \
{float(spec['song_seconds']):.0f} seconds long; use that as "seconds".
THE USER'S DIRECTION FOR THE VIDEO (the most important instruction: follow it exactly and \
literally; it decides what kind of video this is, who we see, where and what happens): \
{direction or "(none given: choose a warm, simple idea that fits the lyrics)"}
{on_screen(spec)}
Split the song into scenes that follow its sections (intro, verses, choruses, bridge, \
instrumental parts, outro). Spread the sections over the whole song length, including intro, \
instrumental breaks and outro where nothing is sung (give those scenes text "(instrumental)")."""
    given = parse_cast(spec.get("cast") or "")
    if given:
        cast_rule = ("- THE CAST IS GIVEN (below). Use these characters with these exact names "
                     "and copy each look word for word. Only add a character who is clearly in "
                     "the text but missing from the list.\n\nTHE CAST:\n" +
                     "\n".join(f"{c['name']}: {c['look']}" for c in given))
    elif nobody:
        cast_rule = "- There are no recurring characters: return an empty characters list."
    else:
        cast_rule = ("- List every recurring character who appears on screen with ONE fixed "
                     "look, detailed enough that an image AI draws the same character every time "
                     "(no changes between scenes unless the story says so). If the direction "
                     "means there is no main character, return an empty list.")
    cast_rule += ("\n- Each character is ONE individual. Groups and crowds (classmates, the "
                  "meadow animals, a choir) are not characters: describe them inside the shots. "
                  "A look never mentions another character, not even for size.")
    prompt = f"""{task}

Then:
{cast_rule}
- ALWAYS begin with an INTRO scene (section "Intro") of opening pictures before any words: \
establish the world and the place, and introduce the main character(s) if there are any. For a \
song it covers the instrumental intro (about 8-15 seconds if you can't tell). For a story it is \
about 16 seconds of opening pictures before the narration starts; include it in the total.
- Write "approach": the direction turned into plain rules for every shot.
- Split it into scenes in order, each with its setting described the same way every time it \
returns, the exact text it covers, and how many seconds it lasts. The scene seconds must add up \
to the total.

The art style chosen for every picture is "{spec.get('style_name') or 'the chosen style'}" (it \
is added to each prompt automatically). Never describe a different medium or style (for \
example don't call it 3D if the style is pencil or watercolor), and keep the art style out of \
character looks.

Title given by the user (keep it if present): {spec.get('title') or '(none)'}

THE {"STORY" if spec["kind"] == "story" else "LYRICS"}:
{text[:60000]}"""
    if progress:
        progress(0.05, f"{C.ai_name()} is reading the {'story' if spec['kind'] == 'story' else 'lyrics'}…")
    res = C.ask(prompt, PLAN, system=SYSTEM, timeout=900)
    if not res.get("scenes"):
        raise StoryError("The AI couldn't split that into scenes. Check the text and try again.")
    if nobody and not given:                     # "no main character" means none, whatever the AI says
        res["characters"] = []
    for c in res.get("characters", []):          # the user's own words always win
        for g in given:
            if g["name"].lower() == c["name"].strip().lower():
                c["name"], c["look"] = g["name"], g["look"]
    for g in given:
        if not any(g["name"].lower() == c["name"].lower() for c in res.get("characters", [])):
            res.setdefault("characters", []).append({"name": g["name"], "look": g["look"],
                                                     "short": short_look(g["look"]), "voice": ""})
    for c in res.get("characters", []):
        c["short"] = (c.get("short") or "").strip() or short_look(c["look"])
    if spec["kind"] == "music":
        res["seconds"] = float(spec["song_seconds"])
    else:
        res["seconds"] = max(10.0, float(res.get("seconds") or words / WORDS_PER_SEC))
    first = res["scenes"][0]
    if not re.search(r"intro|opening", f"{first.get('section', '')} {first['title']}", re.I):
        intro = 2 * int(spec.get("clip") or 8)       # the AI forgot the intro: add one
        res["scenes"].insert(0, {
            "title": "Opening", "section": "Intro", "place": first["place"],
            "text": "(opening pictures before the words begin)", "seconds": intro,
            "mood": first.get("mood", ""),
            "concept": "Establish the world and the place, and introduce the main character "
                       "before the first words."})
        if spec["kind"] == "story":
            res["seconds"] += intro
    # scene lengths must fill the whole video
    total = sum(max(1.0, float(s["seconds"])) for s in res["scenes"])
    for s in res["scenes"]:
        s["seconds"] = max(1.0, float(s["seconds"])) * res["seconds"] / total
    return res


def short_look(look, words=20):
    """A character's look cut to its first ~20 words (when the AI didn't give a short one)."""
    w = (look or "").split()
    if len(w) <= words:
        return " ".join(w).rstrip(".")
    cut = " ".join(w[:words])
    return (cut.rsplit(",", 1)[0] if "," in cut else cut).rstrip(".,")


def parse_cast(text):
    """Characters typed by the user, one per line: "Name: what they look like" (also accepts
    "Name - look"). Lines without a name continue the character above."""
    cast = []
    for line in (text or "").splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line:
            continue
        m = re.match(r"^([^:]{1,40}?)(?:\s*:\s*|\s+[-–]\s+)(.+)$", line)
        if m:
            cast.append({"name": m.group(1).strip(), "look": m.group(2).strip()})
        elif cast:
            cast[-1]["look"] += " " + line
    return cast


CAST = {"type": "object", "properties": {"characters": {"type": "array", "items": CHAR}},
        "required": ["characters"], "additionalProperties": False}


def suggest_cast(spec):
    """The AI describes every character for the user to check and edit before building."""
    import ss_claude as C
    text = (spec.get("text") or "").strip()
    if not text:
        raise StoryError("Paste or load the " + ("story script" if spec.get("kind") == "story"
                                                  else "song lyrics") + " first.")
    given = parse_cast(spec.get("cast") or "")
    if spec.get("onscreen") in ("none", "nopeople") and not given:
        return []                                # no recurring characters wanted
    kept = ""
    if given:
        kept = ("Characters the user already described (keep their words, fill in anything "
                "missing, then add the others):\n" +
                "\n".join(f"{c['name']}: {c['look']}" for c in given))
    style = f"Art style: {spec['style_name']}." if spec.get("style_name") else ""
    direction = (f"The user's direction (follow it): {spec['direction']}"
                 if spec.get("direction") else "")
    what = "story" if spec.get("kind") == "story" else "music video"
    res = C.ask(f"""Describe every recurring character who will appear on screen in this {what} \
(if the direction means there is no main character, return an empty list), so an \
image AI draws each one exactly the same in every picture. For each: name, and one fixed look in \
a single paragraph (species and age, build, face, eyes, skin or fur, hair colour/length/style, \
outfit with colours and materials, shoes, accessories). Keep looks simple and distinctive so they \
are easy to keep consistent, family-friendly, and fitting the art style, but don't write the art \
style itself into the look (it is added to every prompt separately). Each character is ONE \
individual: groups and crowds (classmates, the animals, a choir) are not characters. A look never \
mentions or compares to another character, not even for size (say "about one metre tall", not \
"as tall as the girl"), because each look is also used for a picture of that character alone.
{style}
{direction}
{kept}

{"STORY" if spec.get("kind") == "story" else "LYRICS"}:
{text[:40000]}""", CAST, system=SYSTEM, timeout=600)
    return res.get("characters", [])


IDEAS = {
    "type": "object",
    "properties": {"ideas": {"type": "array", "items": {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "2-5 words"},
                       "direction": {"type": "string", "description": "3-5 sentences: who we "
                                     "see, where, what happens from start to end, and the "
                                     "big visual moment for the chorus"}},
        "required": ["name", "direction"], "additionalProperties": False}}},
    "required": ["ideas"], "additionalProperties": False,
}


def brainstorm(lyrics, notes="", style_name="", progress=None, onscreen="auto"):
    """A few different directions for a music video, from the lyrics."""
    import ss_claude as C
    if not (lyrics or "").strip():
        raise StoryError("Paste the song lyrics first, then brainstorm.")
    if onscreen in ("none", "nopeople"):
        mix = ("Every idea has NO main character: " + (ON_NOPEOPLE if onscreen == "nopeople"
                                                       else ON_NONE) + " Make the 4 ideas "
               "clearly different in setting, journey and visual thread.")
    else:
        mix = ("Mix approaches: a little story with characters that follows the lyrics, a "
               "B-roll video with no main character (different people of all ages and nature "
               "that fit each line), a journey through changing places, and one surprising "
               "idea." + (" The video follows main characters." if onscreen == "main" else ""))
    res = C.ask(f"""Suggest 4 clearly different directions for a music video of this song. \
{mix} Keep every idea warm, wholesome, closely matched to what the lyrics are about, and easy \
for image and video AI to make (simple settings, clear actions). Each idea needs one visual \
thread that ties the video together and a big, memorable image for the chorus.
{f"Art style that will be used: {style_name}." if style_name else ""}
{f"What the user is thinking of so far (build on it): {notes}" if notes else ""}

LYRICS:
{lyrics[:20000]}""", IDEAS, system=SYSTEM, timeout=600)
    return res.get("ideas", [])[:6]


# ---------------------------------------------------------------------------
# step 2: the shots
# ---------------------------------------------------------------------------
PICTURE_RULES = ("the TEXT-TO-IMAGE prompt, SHORT: 1-3 sentences, about 25-50 words. Start "
                 "with the shot size and angle (wide shot, medium shot, close-up, overhead, "
                 "low angle...), then who or what is in it and what they are doing at the START "
                 "of the clip (a recurring character by their short look from the cast list, "
                 "word for word; anyone else by a few words of age, hair and clothing), then the "
                 "place and the light. Concrete and specific, no filler words. Image AIs don't "
                 "know names, so never use names. Describe what IS there; never write 'no ...'. "
                 "No words, signs or letters in the picture. Don't add the art style or the "
                 "picture shape; they are added automatically. " + RULES_LINE)
MOTION_RULES = ("the IMAGE-TO-VIDEO prompt for MiniMax Hailuo, SHORT: 1-2 sentences, about "
                "15-35 words. The subject first and one clear, visible action that fits the clip "
                "and the words, plus a touch of natural movement (hair, leaves, water, light), "
                "then the camera move in plain words. One continuous shot, only what is already "
                "in the picture, believable motion (children move like real children), never "
                "write 'no ...'.")
SHOT = {
    "type": "object",
    "properties": {
        "scene": {"type": "integer", "description": "scene number, from 1"},
        "extra": {"type": "boolean", "description": "true for an extra shot (close-up, "
                  "cutaway, reaction, detail) that the timing doesn't strictly need"},
        "words": {"type": "string", "description": "the narration or lyrics heard during this "
                  "shot, word for word; for intro, instrumental or extra shots a short note "
                  "like \"Gentle musical opening\""},
        "characters": {"type": "array", "items": {"type": "string"},
                       "description": "names of the characters visible in the picture"},
        "title": {"type": "string", "description": "a short shot name, 2-6 words"},
        "purpose": {"type": "string", "description": "the shot description: one sentence on "
                    "what this shot is for in the story"},
        "picture": {"type": "string", "description": PICTURE_RULES},
        "motion": {"type": "string", "description": MOTION_RULES},
        "camera": {"type": "string", "enum": CAMERA,
                   "description": "the MiniMax camera command matching the camera move"},
        "camera2": {"type": "string", "enum": [""] + CAMERA,
                    "description": "optional second move at the same time, or empty"},
    },
    "required": ["scene", "extra", "title", "purpose", "words", "characters", "picture",
                 "motion", "camera", "camera2"],
    "additionalProperties": False,
}
SHOTS = {"type": "object", "properties": {"shots": {"type": "array", "items": SHOT}},
         "required": ["shots"], "additionalProperties": False}


def shot_counts(plan, clip):
    """Shots each scene needs (one per clip), and how many extras."""
    # main shots fill the length exactly (one per clip), shared out by each scene's length
    exact = [s["seconds"] / clip for s in plan["scenes"]]
    want = max(len(exact), math.ceil(plan["seconds"] / clip - 0.1))
    need = [max(1, int(x)) for x in exact]
    left = want - sum(need)
    for i in sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))[:max(0, left)]:
        need[i] += 1
    total = sum(need)
    extras = max(2, math.ceil(total * EXTRA))
    # extras go to the longest scenes first
    order = sorted(range(len(need)), key=lambda i: -plan["scenes"][i]["seconds"])
    add = [0] * len(need)
    for k in range(extras):
        add[order[k % len(order)]] += 1
    return need, add


def make_shots(spec, plan, progress=None, cancel=None, per_request=24):
    import ss_claude as C
    clip = int(spec["clip"])
    need, add = shot_counts(plan, clip)
    cast = "\n".join(f"- {c['name']}: short look to use in prompts: \"{c['short']}\" "
                     f"(full look, for reference: {c['look']}; {c.get('voice', '')})"
                     for c in plan["characters"])
    direction = (spec.get("direction") or "").strip()
    rules = (f"THE USER'S DIRECTION (follow it exactly): {direction}\n" if direction else "") + \
        f"THE APPROACH FOR EVERY SHOT: {plan.get('approach', '')}\n" + \
        (on_screen(spec) if spec["kind"] == "music" else "")
    # groups of scenes, about per_request shots each
    groups, cur, n = [], [], 0
    for i, s in enumerate(plan["scenes"]):
        k = need[i] + add[i]
        if cur and n + k > per_request:
            groups.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += k
    if cur:
        groups.append(cur)
    shots = []
    for gi, grp in enumerate(groups):
        if cancel and cancel():
            raise StoryError("Stopped.")
        if progress:
            progress(0.15 + 0.8 * gi / len(groups),
                     f"{C.ai_name()} is planning shots for scene {grp[0] + 1}"
                     f"{f' to {grp[-1] + 1}' if len(grp) > 1 else ''} of {len(plan['scenes'])}…")
        scenes = "\n\n".join(
            f"SCENE {i + 1}: {plan['scenes'][i]['title']} ("
            f"{plan['scenes'][i].get('section', '')}, {plan['scenes'][i]['seconds']:.0f} s, "
            f"mood: {plan['scenes'][i]['mood']})\n"
            f"Concept: {plan['scenes'][i].get('concept', '')}\n"
            f"Setting: {plan['scenes'][i]['place']}\n"
            f"Needs {need[i]} main shot(s) of {clip} s each, plus {add[i]} extra shot(s).\n"
            f"Text: {plan['scenes'][i]['text']}" for i in grp)
        before = ""
        if shots:
            last = shots[-3:]
            before = ("\n\nShots already planned (never repeat these ideas; vary from them):\n" +
                      "; ".join(s.get("title", "") for s in shots[:-3]) +
                      "\nThe last shots just before these (continue smoothly):\n" +
                      "\n".join(f"- {s['picture']}" for s in last))
        prompt = f"""{"Story" if spec["kind"] == "story" else "Music video"}: \
"{plan['title']}". {plan['summary']}

{rules}

Each main shot is one {clip}-second clip. Main shots go in order and together cover all of each \
scene's text, split evenly so each shot's words fit about {clip} seconds, and each one shows what \
its words are about. Put each scene's extra shots right after the main shot they would cut away \
from; extras are different angles, details or moments, never a copy of another shot.

CAST (in the text-to-image prompt, describe a visible character by their short look, word for \
word):
{cast or "(no recurring characters)"}

LOOK OF THE WHOLE VIDEO: {plan.get('visual_rules', '')}
ART STYLE: "{spec.get('style_name', '')}" is added to every picture automatically; never write a \
different medium or style into the prompts (no "3D" unless the style is 3D).

{scenes}{before}

Plan exactly the number of main and extra shots asked for in each scene. Keep it varied: mix \
wide shots, medium action shots, close-ups, details and unusual angles (overhead, low, through \
a doorway or leaves), with never more than two close-ups or head shots in a row. Choose the \
camera move that suits each moment: calmer moves ([Static shot], [Push in], [Pan left/right]) in \
verses and quiet parts, livelier ones ([Tracking shot], [Truck left/right], [Pedestal up], \
[Pull out]) in choruses and big moments. Keep both prompts short."""
        res = C.ask(prompt, SHOTS, system=SYSTEM, timeout=1200)
        for s in res.get("shots", []):
            s["scene"] = min(max(1, int(s.get("scene") or grp[0] + 1)), len(plan["scenes"]))
            shots.append(s)
    return shots


# ---------------------------------------------------------------------------
# putting the prompts together
# ---------------------------------------------------------------------------
def picture_prompt(shot, plan, style, aspect):
    """The picture description, then the art style: short (the usual) or the full style text.
    Shot lists made before 1.2 have no "length" and keep the full style."""
    if style.get("length", "full") == "full":
        return f"{shot['picture'].strip()}\n\nArt style: {style_block(style, aspect)}"
    return f"{shot['picture'].strip().rstrip('.')}. {style_tag(style, aspect)}"


def minimax_prompt(shot):
    cams = [shot["camera"]] + ([shot["camera2"]] if shot.get("camera2") and
                               shot["camera2"] != shot["camera"] else [])
    return f"{shot['motion'].strip()} [{','.join(cams)}]"[:2000]


def sheet_prompt(char, style, aspect):
    """The character prompt: one full-body reference picture to make first, then attach to
    every shot the character is in."""
    return (f"Full-body character reference image of {char['name']}: "
            f"{char['look'].rstrip('.')}."
            + f" {char['name']} is completely alone in the picture: one single character, with "
            "nobody and nothing else beside them. Standing upright in a relaxed three-quarter "
            "pose, both hands visible, feet fully visible, the complete character from head to "
            "toe. Plain warm ivory studio background, centered composition, a clean consistent "
            f"character design reference.\n\nArt style: {style_block(style, aspect)}")


# ---------------------------------------------------------------------------
# all of it
# ---------------------------------------------------------------------------
def build(spec, progress=None, cancel=None):
    say = progress or (lambda f, m: None)
    spec = dict(spec)
    if spec.get("kind") not in ("story", "music"):
        raise StoryError("Choose Music video or Story first.")
    if not (spec.get("text") or "").strip():
        raise StoryError("Paste or load the " + ("story script." if spec["kind"] == "story"
                                                  else "song lyrics."))
    if spec["kind"] == "story":
        spec["clip"] = 8
    else:
        spec["clip"] = int(spec.get("clip") or 8)
        if not spec.get("song_seconds") or float(spec["song_seconds"]) < 5:
            raise StoryError("Enter how long the song is (for example 3:15).")
    style = dict(spec.get("style") or find_style(spec.get("style_id")))
    style["wording"] = "original" if spec.get("wording") == "original" else "improved"
    style["length"] = "full" if spec.get("style_len") == "full" else "short"
    aspect = spec.get("aspect") or "16:9"
    spec["style_name"] = style.get("name", "")
    if style["length"] == "short" and not style.get("style_short"):
        try:                                     # one of the user's own styles: shorten it once
            say(0.02, "Writing a short version of your art style…")
            style["style_short"] = shorten_style(style)
            remember_short(style.get("id", ""), style["style_short"])
        except Exception as ex:                  # the first few phrases will do instead
            log("short style skipped", ex)
    plan = read_script(spec, say)
    if cancel and cancel():
        raise StoryError("Stopped.")
    shots = make_shots(spec, plan, say, cancel)
    # times: main shots one clip each, in order; extras sit at the shot they cut away from
    t, n = 0.0, 0
    for s in shots:
        n += 1
        s["n"] = n
        if s.get("extra"):
            s["start"] = max(0.0, t - spec["clip"])
            s["end"] = t
        else:
            s["start"], s["end"] = t, t + spec["clip"]
            t += spec["clip"]
        s["picture_prompt"] = picture_prompt(s, plan, style, aspect)
        s["minimax_prompt"] = minimax_prompt(s)
    for c in plan["characters"]:
        c["sheet_prompt"] = sheet_prompt(c, style, aspect)
    main = [s for s in shots if not s.get("extra")]
    for i, sc in enumerate(plan["scenes"]):                # each scene's place on the clock
        mine = [s for s in shots if s["scene"] == i + 1 and not s.get("extra")]
        sc["start"] = mine[0]["start"] if mine else 0.0
        sc["end"] = mine[-1]["end"] if mine else 0.0
        sc["clips"] = len(mine)
    result = {"kind": spec["kind"], "title": plan["title"], "summary": plan["summary"],
              "seconds": plan["seconds"], "clip": spec["clip"], "aspect": aspect,
              "visual_rules": plan.get("visual_rules", ""), "approach": plan.get("approach", ""),
              "style": {k: style.get(k, "") for k in ("id", "name", "style", "clean",
                                                      "style_opt", "clean_opt", "style_short",
                                                      "wording", "length")},
              "characters": plan["characters"], "scenes": plan["scenes"], "shots": shots,
              "main_shots": len(main), "extra_shots": len(shots) - len(main),
              "covers": len(main) * spec["clip"], "made": time.strftime("%Y-%m-%d %H:%M"),
              "spec": {k: spec.get(k) for k in ("kind", "title", "song_seconds", "clip",
                                                 "style_id", "aspect", "direction", "cast",
                                                 "wording", "onscreen", "style_len")}}
    add_extras(result, style, say)
    say(0.97, "Saving the shot list…")
    result["files"] = save(result)
    say(1.0, f"Done: {len(main)} shots plus {len(shots) - len(main)} extras.")
    return result


def out_dir():
    d = Path(documents_dir()) / "StoryStudio" / "Shot lists"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(r):
    d = out_dir()
    base = safe_name(r["title"])[:80]
    txt, js = d / f"{base} - shot list.txt", d / f"{base} - shot list.json"
    js.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf8")
    txt.write_text(as_text(r), encoding="utf8")
    return {"text": str(txt), "json": str(js), "folder": str(d)}


def mmss(sec):
    sec = int(round(float(sec)))
    return f"{sec // 60}:{sec % 60:02d}"


def as_text(r):
    kind = "Story" if r["kind"] == "story" else "Music video"
    L = [f"{r['title']}  ({kind})", "=" * 60, r["summary"], ""]
    if r["spec"].get("direction"):
        L += ["Direction: " + r["spec"]["direction"], ""]
    on = {"main": "Main characters", "none": "No main character (B-roll)",
          "nopeople": "No people"}.get(r["spec"].get("onscreen") or "")
    if on:
        L += ["Who's in it: " + on, ""]
    L += [f"Length: about {mmss(r['seconds'])}   Clips: {r['clip']} seconds each   "
          f"Shape: {r['aspect']}   Art style: {r['style']['name']}",
          f"Shots: {r['main_shots']} main + {r['extra_shots']} extra "
          f"(the main shots cover {mmss(r['covers'])})", ""]
    if r.get("visual_rules"):
        L += ["Keeping it one world: " + r["visual_rules"], ""]
    if r.get("title_image"):
        L += ["TITLE IMAGE (transparent background)", "-" * 60,
              "Make it in ChatGPT and ask for a transparent background (PNG). Nano Banana can't "
              "make true transparency, so remove the background afterwards there.", "",
              r["title_image"]["prompt"], ""]
    if r.get("thumbnail"):
        L += ["YOUTUBE THUMBNAIL", "-" * 60, r["thumbnail"]["prompt"], ""]
    if r.get("suno"):
        L += ["BACKGROUND THEME MUSIC (Suno)", "-" * 60]
        for m in r["suno"]:
            L += [f"{m['name']}:", m["prompt"], ""]
    if r["characters"]:
        L += ["CHARACTERS", "-" * 60,
              "Make each character's picture first, then attach it to every shot they appear in "
              "(ChatGPT: upload it with the prompt; Nano Banana: add it as a reference image and "
              "say \"use this picture for the character's look\").", ""]
        for c in r["characters"]:
            L += [f"{c['name']}: {c['look']}", "", "Character prompt:", c["sheet_prompt"], ""]
    scene = None
    L += ["SHOTS", "-" * 60]
    for s in r["shots"]:
        if s["scene"] != scene:
            scene = s["scene"]
            sc = r["scenes"][scene - 1]
            L += ["", f"SCENE {scene}: {sc['title'].upper()}",
                  f"{sc.get('section', '')} · {mmss(sc.get('start', 0))}-{mmss(sc.get('end', 0))}"
                  f" · {sc.get('clips', 0)} clips", f"Scene concept: {sc.get('concept', '')}", ""]
        tag = "EXTRA shot" if s.get("extra") else "Shot"
        L.append(f"{tag} {s['n']}: {s.get('title', '')}   {mmss(s['start'])}-{mmss(s['end'])}"
                 + (f"   ({', '.join(s['characters'])})" if s.get("characters") else ""))
        if s.get("words"):
            L.append(f"  Words: {s['words']}")
        if s.get("purpose"):
            L.append(f"  What this shot is for: {s['purpose']}")
        L += ["", "  Text-to-image prompt (ChatGPT / Nano Banana):",
              "  " + s["picture_prompt"].replace("\n", "\n  "), "",
              f"  Image-to-video prompt (MiniMax, {r['clip']}-second clip):",
              "  " + s["minimax_prompt"], ""]
    L += ["", "TIPS", "-" * 60,
          "- Attach each character's picture to every shot they're in, so they stay the same.",
          "- Each image-to-video prompt is one continuous clip: no cuts inside a clip.",
          "- Extra shots are close-ups and cutaways to use wherever an edit needs one.",
          "- The timing is a plan: line the cuts up with the real narration or song when "
          "editing (StoryStudio's Edit feedback can check the beat)."]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# extras: Suno theme music, YouTube thumbnail, the title lettering
# ---------------------------------------------------------------------------
EXTRAS = {
    "type": "object",
    "properties": {
        "suno": {"type": "array", "description": "3 different background theme music prompts",
                 "items": {"type": "object", "properties": {
                     "name": {"type": "string", "description": "2-4 words, e.g. 'Gentle "
                              "wonder theme'"},
                     "prompt": {"type": "string", "description": "a Suno style prompt, under "
                                "200 characters: genre, instruments, tempo in bpm, feel, and "
                                "end with 'instrumental, no vocals'. Background music that sits "
                                "under narration: gentle, not busy."}},
                     "required": ["name", "prompt"], "additionalProperties": False}},
        "thumbnail": {"type": "string", "description": "the YouTube thumbnail picture, 3-5 "
                      "sentences: the single most eye-catching moment of the video (with no "
                      "main character, the most striking image instead: people or nature), any "
                      "main character described by their look with a big clear expression, a "
                      "simple bold composition readable at small size, bright contrasting "
                      "colours, and empty space on one side where the title will sit"},
        "lettering": {"type": "string", "description": "a few words on the title lettering "
                      "that fit the video's feel, e.g. 'bold rounded cheerful lettering' for a "
                      "fun kids' song, 'elegant warm serif lettering' for a tender song"},
        "title_look": {"type": "string", "description": "how the title lettering should look, "
                       "2-3 sentences: letter shapes, colours, materials or textures, and small "
                       "decorations taken from the story (e.g. leaves, stars), fitting the art "
                       "style and readable for children"},
    },
    "required": ["suno", "thumbnail", "lettering", "title_look"],
    "additionalProperties": False,
}


def extras(r, progress=None):
    """Suno prompts (stories), a thumbnail prompt, and a title image prompt (stories)."""
    import ss_claude as C
    if progress:
        progress(0.94, "Writing the music, thumbnail and title prompts…")
    cast = "\n".join(f"- {c['name']}: {c['look']}" for c in r["characters"])
    scenes = "\n".join(f"- {s['title']}: {s.get('concept', '')} (mood: {s.get('mood', '')})"
                       for s in r["scenes"])
    res = C.ask(f"""For the {'story' if r['kind'] == 'story' else 'music video'} \
"{r['title']}": {r['summary']}
{f"The approach: {r['approach']}" if r.get('approach') else ""}
Scenes:
{scenes}
Cast:
{cast or '(none)'}
Art style: {r['style']['name']} (added to the picture prompts automatically; never describe a \
different medium or style)

Write: three Suno prompts for gentle instrumental background theme music that follows the feel \
of the story (each a different take: for example a warm main theme, a lighter playful version, \
a tender emotional version); the YouTube thumbnail picture; and how the title lettering should \
look.""", EXTRAS, system=SYSTEM, timeout=600)
    return res


def thumbnail_prompt(desc, title, style, aspect="16:9", lettering=""):
    letters = (lettering or "large, clear lettering that fits the feel of the video").strip()
    return (f"{desc.strip()} Bold, eye-catching YouTube thumbnail composition with the title "
            f"\"{title}\" in {letters.rstrip('.')}, large in the empty space and easy to read at "
            f"a small size.\n\nArt style: {style_text(style)} 16:9 landscape composition, "
            "high-resolution, crisp faces and details, clean sharp edges.")


def title_prompt(look, title, style):
    return (f"The title \"{title}\" as standalone title lettering on a fully transparent "
            f"background (PNG with transparency). {look.strip()} Only the title lettering and "
            "its small decorations, centered, with clean crisp edges and nothing behind it."
            f"\n\nArt style: {style_text(style)}")


def add_extras(r, style, progress=None):
    try:
        x = extras(r, progress)
    except StoryError as ex:
        log("script extras skipped", ex)
        return
    r["thumbnail"] = {"desc": x["thumbnail"], "lettering": x.get("lettering", ""),
                      "prompt": thumbnail_prompt(x["thumbnail"], r["title"], style,
                                                 lettering=x.get("lettering", ""))}
    if r["kind"] == "story":
        r["suno"] = x["suno"][:3]
        r["title_image"] = {"look": x["title_look"],
                            "prompt": title_prompt(x["title_look"], r["title"], style)}


# ---------------------------------------------------------------------------
# a new version of one prompt
# ---------------------------------------------------------------------------
REDO = {"type": "object", "properties": {"text": {"type": "string"},
                                         "short": {"type": "string"},
                                         "camera": {"type": "string", "enum": CAMERA},
                                         "camera2": {"type": "string", "enum": [""] + CAMERA}},
        "required": ["text", "short", "camera", "camera2"], "additionalProperties": False}
NO_SHORT = ' Set "short" to "".'
REDO_SHOT = {"type": "object", "properties": {
    "title": {"type": "string"}, "picture": {"type": "string"}, "motion": {"type": "string"},
    "characters": {"type": "array", "items": {"type": "string"}},
    "camera": {"type": "string", "enum": CAMERA},
    "camera2": {"type": "string", "enum": [""] + CAMERA}},
    "required": ["title", "picture", "motion", "characters", "camera", "camera2"],
    "additionalProperties": False}


FIXED = {"type": "object", "properties": {"shots": {"type": "array", "items": {
    "type": "object", "properties": {"n": {"type": "integer"}, "picture": {"type": "string"}},
    "required": ["n", "picture"], "additionalProperties": False}}},
    "required": ["shots"], "additionalProperties": False}


def restyle_shots(r, c, old_look, style, progress=None, per_request=12):
    """After a character's look changes, rewrite the text-to-image prompt of every shot they
    are in so it describes the new look (nothing else changes)."""
    import ss_claude as C
    mine = [s for s in r["shots"] if any(x.lower() == c["name"].lower()
                                          for x in s.get("characters", []))]
    new_look = f"{c['short']} (in full: {c['look']})" if c.get("short") else c["look"]
    done = 0
    for k in range(0, len(mine), per_request):
        part = mine[k:k + per_request]
        if progress:
            progress(0.2 + 0.75 * k / max(1, len(mine)),
                     f"Updating {c['name']} in shots {part[0]['n']} to {part[-1]['n']}…")
        listing = "\n\n".join(f"SHOT {s['n']}: {s['picture']}" for s in part)
        res = C.ask(f"""The character {c['name']} has a NEW look. In each text-to-image prompt \
below, change only how {c['name']} is described so it matches the new look exactly (hair, eyes, \
age, outfit and colours). Keep everything else word for word: the framing, the action, the \
other characters, the setting, the light. Characters are described by their look, never by name. \
Use the new short look word for word, and keep each prompt as short as it is now.

OLD LOOK: {old_look}
NEW LOOK: {new_look}

{listing}

Return every shot with its number and the updated prompt in "picture".""", FIXED, system=SYSTEM,
                    timeout=600)
        new = {int(x["n"]): x["picture"].strip() for x in res.get("shots", [])}
        for s in part:
            if new.get(s["n"]):
                s["picture"] = new[s["n"]]
                s["picture_prompt"] = picture_prompt(s, r, style, r["aspect"])
                done += 1
    return done


def redo(r, n, which, note="", progress=None):
    """Write a new text-to-image ("picture") or image-to-video ("motion") prompt for shot n,
    or a new look for a character (which="character", n = index). Saves the shot list."""
    import ss_claude as C
    style = r["style"] if r["style"].get("style") else find_style(r["style"]["id"])
    about = (f"the {'story' if r['kind'] == 'story' else 'music video'} \"{r['title']}\": "
             f"{r['summary']}")
    spec = r.get("spec") or {}
    rules = ((f"THE USER'S DIRECTION (follow it exactly): {spec['direction']}\n"
              if spec.get("direction") else "") +
             (f"THE APPROACH FOR EVERY SHOT: {r['approach']}\n" if r.get("approach") else "") +
             (on_screen(spec) + "\n" if r["kind"] == "music" and spec.get("onscreen") else ""))
    change = f"What the user wants changed: {note}" if note else "Make a fresh, better version."
    if which in ("suno", "thumbnail", "title"):
        if which == "suno":
            cur = r["suno"][int(n)]
            ask = (f"Write a new Suno style prompt for gentle instrumental background theme "
                   f"music for {about} Current one ({cur['name']}): {cur['prompt']}\n{change}\n"
                   "Under 200 characters: genre, instruments, tempo in bpm, feel, ending with "
                   "'instrumental, no vocals'.")
        elif which == "thumbnail":
            ask = (f"Write a new YouTube thumbnail picture description for {about} Current: "
                   f"{r['thumbnail']['desc']}\n{change}\nRules: "
                   f"{EXTRAS['properties']['thumbnail']['description']}")
        else:
            ask = (f"Describe a new look for the title lettering \"{r['title']}\" for {about} "
                   f"Current: {r['title_image']['look']}\n{change}\nRules: "
                   f"{EXTRAS['properties']['title_look']['description']}")
        res = C.ask(ask + '\nPut it in "text"; set camera to "Static shot" and camera2 to "".' +
                    NO_SHORT, REDO, system=SYSTEM, timeout=300)
        text = res["text"].strip()
        if which == "suno":
            r["suno"][int(n)]["prompt"] = text
            out = {"suno": r["suno"]}
        elif which == "thumbnail":
            lettering = r["thumbnail"].get("lettering", "")
            r["thumbnail"] = {"desc": text, "lettering": lettering,
                              "prompt": thumbnail_prompt(text, r["title"], style,
                                                         lettering=lettering)}
            out = {"thumbnail": r["thumbnail"]}
        else:
            r["title_image"] = {"look": text, "prompt": title_prompt(text, r["title"], style)}
            out = {"title_image": r["title_image"]}
        r["files"] = save(r)
        return out
    if which == "character":
        c = r["characters"][int(n)]
        res = C.ask(f"""Write a new fixed look for {c['name']} in this \
{'story' if r['kind'] == 'story' else 'music video'} ("{r['title']}"): one paragraph with \
species and age, build, face, eyes, skin or fur, hair colour/length/style, outfit with colours \
and materials, shoes, accessories. ONE individual only. Never mention or compare to another \
character, not even for size (say "about one metre tall", never "as tall as the girl"); don't \
mention the art style or medium. Current look: {c['look']}
{f"What to change: {note}" if note else "Make it a fresh, appealing take that still fits."}
Put the look in "text", and the same look in 12-20 words (the key features that identify them at \
a glance: age or species, hair, skin, main outfit colours) in "short"; set camera to \
"Static shot" and camera2 to "".""", REDO, system=SYSTEM, timeout=300)
        old = f"{c['short']} (in full: {c['look']})" if c.get("short") else c["look"]
        c["look"] = res["text"].strip()
        c["short"] = res.get("short", "").strip() or short_look(c["look"])
        c["sheet_prompt"] = sheet_prompt(c, style, r["aspect"])
        changed = restyle_shots(r, c, old, style, progress)
        r["files"] = save(r)
        return {"character": c, "shots_changed": changed}
    shot = next(s for s in r["shots"] if s["n"] == int(n))
    sc = r["scenes"][shot["scene"] - 1]
    cast = "\n".join(f"- {c['name']}: short look \"{c.get('short') or c['look']}\""
                     for c in r["characters"])
    idx = r["shots"].index(shot)
    near = "\n".join(f"- shot {s['n']}: {s['picture'][:200]}" for s in
                     r["shots"][max(0, idx - 2):idx + 3] if s is not shot)
    head = f"""{'Story' if r['kind'] == 'story' else 'Music video'} "{r['title']}": {r['summary']}
{rules}
Scene {shot['scene']}: {sc['title']} ({sc.get('section', '')}). {sc.get('concept', '')}
Setting: {sc['place']}
Shot {shot['n']}: {shot.get('title', '')}. {shot.get('purpose', '')}
Words heard: {shot.get('words', '')}
Clip length: {r['clip']} seconds.
CAST (describe a visible character by their short look, word for word):
{cast or '(no recurring characters)'}
Shots next to it (don't repeat them):
{near or '(none)'}"""
    if which == "picture":                       # a whole new shot: the picture and its motion
        res = C.ask(f"""{head}

The user doesn't like this shot and wants a new one:
Picture: {shot['picture']}
Motion: {shot['motion']}
{f"What the user wants: {note}" if note else "Write a new, better shot for the same words: a different idea, framing or subject is welcome."}

Write the new shot: "picture" is {PICTURE_RULES} "motion" is {MOTION_RULES} Set "camera" (and \
"camera2", or "") to the MiniMax camera command matching the motion; "title" is a short shot \
name, 2-6 words; "characters" lists the names of recurring characters visible (or none).""",
                    REDO_SHOT, system=SYSTEM, timeout=300)
        shot["picture"] = res["picture"].strip()
        shot["motion"] = res["motion"].strip()
        shot["title"] = res.get("title", "").strip() or shot.get("title", "")
        shot["characters"] = [x for x in res.get("characters", [])
                              if any(x.lower() == c["name"].lower() for c in r["characters"])]
        shot["camera"], shot["camera2"] = res["camera"], res.get("camera2", "")
        shot["picture_prompt"] = picture_prompt(shot, r, style, r["aspect"])
        shot["minimax_prompt"] = minimax_prompt(shot)
        r["files"] = save(r)
        return {"shot": shot}
    res = C.ask(f"""{head}
The text-to-image prompt (the motion must fit this picture): {shot['picture']}

CURRENT IMAGE-TO-VIDEO PROMPT (the user wants a new version): {shot['motion']}
{f"What the user wants changed: {note}" if note else "Write a fresh, better version for the same picture."}

Write the new prompt in "text", following this rule: {MOTION_RULES}
Set camera (and camera2, or "") to the MiniMax camera command that matches the camera move in \
your new text.""" + NO_SHORT, REDO, system=SYSTEM, timeout=300)
    shot["motion"] = res["text"].strip()
    shot["camera"], shot["camera2"] = res["camera"], res.get("camera2", "")
    shot["minimax_prompt"] = minimax_prompt(shot)
    r["files"] = save(r)
    return {"shot": shot}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def recent(limit=8):
    out = []
    for f in sorted(out_dir().glob("* - shot list.json"), key=lambda p: -p.stat().st_mtime)[:limit]:
        try:
            r = json.loads(f.read_text(encoding="utf8"))
            out.append({"path": str(f), "title": r.get("title", f.stem), "kind": r.get("kind"),
                        "made": r.get("made", ""), "shots": len(r.get("shots", []))})
        except (OSError, ValueError):
            continue
    return out
