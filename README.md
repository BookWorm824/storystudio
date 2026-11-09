# StoryStudio

An AI editing helper for stories, movies and music videos made with AI, for Windows.

- **Script builder**: song lyrics or a story become a shot list with a picture prompt
  (ChatGPT / Nano Banana) and a MiniMax Hailuo video prompt for every shot, plus character,
  Suno, thumbnail and title prompts.
- **Sound effects**: reads the story and the Premiere timeline, watches the movie, and makes
  every sound lined up with the picture at the right levels.
- **Fix the audio**, **karaoke lyrics** and **edit feedback** (notes as markers on the
  Premiere timeline).

Premiere projects are never changed: StoryStudio writes copies.

## Installing

Download `StoryStudio-Setup.zip` from the person who shared it with you, unzip it, and
double-click **Install StoryStudio.bat**. See `installer/READ ME FIRST.txt`.

## Updates

Installed copies update themselves from this repository: `StoryStudio.pyw` and `ss_update.py`
read `version.json` here and download every file listed in `"files"` when its `"version"` is
higher.

Releasing: bump `version` in `version.json`, list every app file in `"files"` (flat names),
install new pip packages at runtime when first needed (and add them to
`installer/install.ps1`), then push to `main`.
