# How to write prompts for MiniMax (Hailuo)

Research for StoryStudio's script builder, 2026-11-07. Sources are listed at the bottom.
"Official" means MiniMax's own API docs or MiniMax's own GitHub prompt guide.

## Which MiniMax model?

**StoryStudio targets Hailuo 2.3 / 02** (confirmed 2026-11-07): 6-second clips, bracket camera commands. The H3 section is kept for later.

MiniMax now has two generations with different prompt styles:

| | Hailuo 2.3 / 2.3 Fast / 02 | MiniMax H3 (Hailuo 3), the newest |
|---|---|---|
| Clip length | 6 s (or 10 s at 768p; 1080p is 6 s only) | 4 to 15 s, any whole number |
| Prompt limit | 2,000 characters | 7,000 characters |
| Camera | 15 bracket commands, e.g. `[Push in]` | Plain sentences ("the camera pushes in slowly") |
| Sound | Silent video | Makes sound, voices and music too |
| Shots | One shot per clip | Several shots per clip, with cut times |

## Rules for both models (image-to-video)

1. **The picture already shows what things look like. The prompt describes what moves.**
   Don't describe the picture again; it adds noise and can cause warping. (Official guide: "Anchor
   Shot 1 on the first frame's style, subjects, composition, and setting, then continue the
   action. Keep identity, clothing, colors, and layout consistent.")
2. **One subject, one clear action, one camera move per clip.** Crowded prompts and changing the
   subject partway through are the main causes of melting faces and shape-shifting.
3. **Name the subject first** so the model knows who moves.
4. **Say what you want, not what you don't.** Negative wording ("no morphing") can add the very
   thing named. (Official ComfyUI guide: banned concepts can add them; state what should
   appear instead.)
5. **Concrete over vague.** "Her hair lifts in the breeze as she turns toward the window" beats
   "cinematic, beautiful, emotional".
6. **Stay inside the frame.** If the picture is a close-up, don't ask for a wide shot; the
   model has to invent the rest and it goes wrong.
7. **Gentle motion for kids' animation.** Slow, small camera moves and simple actions keep
   faces stable. Fast motion blurs.
8. **Match the picture to the video shape** (16:9 picture for a 16:9 video).

## Hailuo 2.3 / 02 format

- Camera commands, typed exactly, inside square brackets:
  `[Truck left]` `[Truck right]` `[Pan left]` `[Pan right]` `[Push in]` `[Pull out]`
  `[Pedestal up]` `[Pedestal down]` `[Tilt up]` `[Tilt down]` `[Zoom in]` `[Zoom out]`
  `[Shake]` `[Tracking shot]` `[Static shot]`
- Several moves at the same time go in one bracket, at most 3: `[Pan left,Pedestal up]`.
- Moves one after another go in order: `... [Push in], then ... [Pull out]`.
- Brackets are more accurate than describing the move in words.
- Put the command right after the part of the description it belongs to.
- Example: `The little girl slowly lifts her head and smiles toward the window as sunlight
  warms her face. [Push in]`

## MiniMax H3 format (MiniMax's official prompt guide)

```
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] 3D CG animated storybook. <what moves, with the camera move written as a sentence>. [Shot 2] At 00:04.000, the camera cuts to <new information>.
overall_soundscape: <1-4 sentences of background and action sounds, or N/A>
non_diegetic_music: <1-3 sentences of instruments, tempo, dynamics, or N/A>
```

- The first line (for a starting picture) and the field names are fixed wording.
- Start shot 1 with a style label (3D CG, 2D-animated, live-action...).
- `[Shot 1]` has no time. Later shots start "At 00:04.000, the camera cuts to...", with times
  going up and staying inside the clip length.
- Only cut for new information (a new place, subject, time or viewpoint). For a closer view, move
  the camera instead.
- Camera moves: push in / pull out, zoom, pan, truck, tilt, pedestal, arc shot, tracking shot,
  static shot, shake (slight/strong), POV, roll. Optional "with small amplitude", "at slow speed".
- Speech: speaker IDs (S1), (S2) and `<d>[English] the exact words</d>`. Narration over the top:
  "says in an off-screen voiceover" and say the character's lips stay closed.
- Music field: instruments, tempo, dynamics; no mood words.
- Timestamps as single points (`At 00:05.000`), never ranges like `[5s-8s]`.

## What this means for StoryStudio

- Each shot gets a picture prompt (ChatGPT / Nano Banana, using the chosen style block) and a
  MiniMax prompt that only describes motion and camera.
- Characters stay the same because the picture carries their look; the MiniMax prompt says
  "keeps the same outfit and hair" only when there's a risk of change.
- Clip length follows the edit: 6 s shots for Hailuo 2.3/02; up to 15 s for H3.
- Storytime videos already have narration and music, so for H3: soundscape and music `N/A`
  (or only small sounds), and no dialogue unless a character speaks on screen.

## Sources

- MiniMax API docs, image-to-video (official): https://platform.minimax.io/docs/api-reference/video-generation-i2v
- MiniMax API docs, video generation guide (official): https://platform.minimax.io/docs/guides/video-generation
- MiniMax's H3 prompt-writing skill (official): https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/SKILL.md
- ComfyUI MiniMax H3 prompt guide: https://docs.comfy.org/tutorials/video/minimax/minimax-h3-prompt-guide
- HeyMarmot H3 format guide: https://heymarmot.com/blog/hailuo-h3-prompting-guide
- Akool Hailuo 2.3 guide: https://akool.com/blog-posts/minimax-hailuo-video-prompt-guide
- Replicate, video-01-director readme: https://replicate.com/minimax/video-01-director/readme
