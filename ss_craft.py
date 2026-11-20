"""The craft of a wordless slapstick cartoon short, in one place.

Distilled from how the classic theatrical cartoon shorts of the 1930s-50s were made (the
animators' own notes on timing, takes and holds; the directors' rules for gags; bar-sheet
timing; the sound-effects and music practice of the period) and from what people making films
with AI pictures and clips struggle with today. Used by the script builder (writing the gags
and the shots), the timing check, the sound plan and the clip check, so they all agree.
Times are in seconds; the classic sources give them in frames at 24 a second.
"""

INTRO = (
    "THIS IS A SILENT CARTOON: a wordless slapstick short in the spirit of the classic "
    "theatrical cartoon shorts of the 1930s to 1950s. No narration, no dialogue, no words or "
    "signs anywhere: everything is told by what we see, and by sound effects and music.\n")

STORY = (
    "STORY AND GAGS\n"
    "- One likeable character wants one small, ordinary thing (a nap, a snack, a ball). One "
    "small nuisance or one prop stands in the way. One place. Set the rules of this little "
    "world in the opening and never break them.\n"
    "- The hero never starts the trouble and is never mean; the nuisance isn't mean either, "
    "it is just being itself. Every plan the hero makes backfires through his own eagerness "
    "or plain gravity. He is embarrassed, never hurt, and bounces back within a second. "
    "Nothing frightening, no weapons, no fire or explosions near anyone.\n"
    "- A gag has two halves: first make the viewer expect something, clearly; then something "
    "else happens. The best gags add a topper: the expected thing lands, and straight after "
    "it a second, smaller surprise.\n"
    "- How many: about 2-3 gags plus a topper in one minute, 4-5 in two minutes, 5-7 in "
    "three. Each one bigger than the last (a try, a bigger try, the biggest try backfires), "
    "never the same gag the same way twice. A running gag returns three times, changed each "
    "time. After a few gags with a wind-up, one that happens with no warning is a gag in "
    "itself.\n"
    "- Rhythm: open slowly, alternate slow 'thinking' gags with quick ones, and pack the last "
    "quarter about twice as tight, building to the biggest gag. Then one calm, warm closing "
    "beat: the hero is fine, a little wiser or happily giving up, and may glance at the "
    "viewer.\n")

ACTING = (
    "ACTING WITHOUT WORDS\n"
    "- THINKING SHOWS. Before every decision give a beat where the hero stops and thinks: the "
    "brows lower, a pause, one brow lifts with a glance to the side. The viewer must be able "
    "to read every thought.\n"
    "- A face changes while the body is still, never in the middle of a big move. Start each "
    "change from the opposite feeling (calm before startled, proud before flattened).\n"
    "- One thing at a time, crystal clear. The eyes go first, then the head, then the body.\n"
    "- The wind-up is the opposite of the action and slower than it: a crouch before the "
    "leap, a lean back before the dash. Slow wind-up, fast action.\n"
    "- The reaction (a 'take'): he sees it, shrinks down, pops up wide-eyed, settles, and "
    "then HOLDS. A slow look straight at the viewer is the classic helpless beat: use it "
    "once or twice, not after every gag.\n"
    "- When a character holds still, something small stays alive: breathing, an ear, the "
    "tail.\n"
    "- Cartoon motion is allowed in the image-to-video prompt: squash and stretch, legs "
    "spinning before a dash, a double take. Name the poses it passes through (crouches low, "
    "springs up long, lands flat with a squash and one bounce).\n")

STAGING = (
    "STAGING THE SHOTS\n"
    "- A physical gag plays in ONE WIDE shot (shot type WS) with a still camera and the whole "
    "gag in the frame, feet included. Never cut closer in the middle of a gag; cut to the "
    "close-up reaction AFTER it lands.\n"
    "- The usual chain for one gag: the setup (wide: where everything is), the thinking or "
    "wind-up (medium or close), the gag itself (wide, one action), the reaction (close-up or "
    "REACT), and sometimes the topper (wide again). Open each scene on an establishing shot, "
    "and use a detail shot (INS) for the prop the gag turns on.\n"
    "- The hero and the nuisance each keep their own side of the screen for the whole "
    "cartoon; say it in every picture prompt ('on the left of the picture, facing right'). "
    "Whoever leaves at the right edge comes into the next shot from the left. Leave more "
    "empty room in front of where a character looks or moves than behind.\n"
    "- For a reaction the face sits near the middle of the frame. Backgrounds stay simple and "
    "soft so the character's shape reads at a glance.\n"
    "- Each picture is the START of its clip: draw the pose just BEFORE the action (the "
    "crouch, not the leap; the reach, not the grab), so the video still has the action to do. "
    "Nothing important happens in the first instant of a clip.\n"
    "- Easy to animate: one clear action per shot, at most two characters in a shot, simple "
    "props. Avoid hands or paws picking up and handling small things; cut to a detail shot "
    "of the thing already moving or already in place instead.\n"
    "- Calm beats look alike (the same soft colours and framing); save the strongest "
    "contrast, the brightest colour and the biggest shapes for the climax.\n")

SOUNDS = (
    "SOUND (in \"words\", list the shot's sound effects in order; never speech)\n"
    "The classic vocabulary: a fall or drop = slide whistle down; a rise or lift = slide "
    "whistle up; a bonk on the head = hollow bonk or anvil clang, then a wobbly cymbal ring "
    "while he sways; slamming into something = dull drum thud; a big crash = clattering pots "
    "and pans; a sudden stop = tyre screech; dashing off = whizz or ricochet zip; a bounce, "
    "spring or poke = boing (higher or lower each time); tiptoeing or sneaking = plucked "
    "strings or xylophone steps; a blink or a small idea = one xylophone note or a ding; a "
    "tug or pluck = string pluck; stuck in something = hollow thunk, then a pop when he comes "
    "free; spinning or dizzy = whirr, then tweeting birds; running on the spot = a quick "
    "patter before the zip; swelling or stretching = rising squeak or rubber stretch; and "
    "splashes, squishes, creaks and other real sounds, played big.\n"
    "Wordless voice sounds count too: gasp, gulp, sniff, yelp, giggle, sigh, snore, a nervous "
    "whistle.\n"
    "Sounds are bigger and sillier than real life and land exactly on the action. Just before "
    "the biggest hit of a gag everything goes quiet for a moment: write \"(silence)\" there.\n")

CUTE = (
    "CUTE CHARACTER DESIGN: a big head (nearly half the body's height), large wide-set eyes "
    "set low on the face, a small nose and mouth, chubby cheeks, short thick legs and a round "
    "tummy; simple rounded shapes that read as a clear outline; big readable expressions. "
    "One small flaw or a streak of mischief keeps cute from being too sweet. Keep the cast to "
    "the hero and one nuisance (a third character only if the idea needs it).")

MUSIC = (
    " This is a SILENT CARTOON, so the music carries the whole short instead of sitting under "
    "narration. Write the three Suno prompts as lively instrumental cartoon scores in the "
    "spirit of classic theatrical cartoon music, each for a different part of the cartoon and "
    "at a clearly different speed, so the music can change from one part to the next: a "
    "calm strolling theme (about 90 bpm), a sneaking or tiptoe theme (60-90 bpm, plucked "
    "pizzicato strings and xylophone), and a chase or finale theme (144-180 bpm). The 'funny' "
    "instruments are the ones at the extremes: tuba, bassoon, trombone, double bass and "
    "piccolo, with clarinet, muted trumpet, xylophone and woodblock. Each should be able to "
    "stop dead for a gag and bounce back.")

# for the timing and pacing check of an edit
TIMING = (
    "This is a SILENT SLAPSTICK CARTOON (no narration). Comic timing is everything, and the "
    "classic shorts worked to these lengths:\n"
    "- Nothing important in the first third of a second after a cut: the eye needs that long "
    "to find the picture. If the action starts on the very first frame, it's missed.\n"
    "- The thinking pause before the hero acts: half a second to a second. Don't trim it "
    "away; it is what makes him a character.\n"
    "- The wind-up (he gets ready and we see it coming): about half a second to a second and "
    "a half, slower than the action that follows.\n"
    "- The gag itself is FAST and plays whole in one wide shot: never cut inside it. If the "
    "bonk, splash or tumble floats, speed it up (130 to 200 percent). Hold about half a "
    "second of stillness just before the payoff, then cut or land it.\n"
    "- The reaction: the take itself lasts about three quarters of a second, then HOLD the "
    "face 1 to 2 seconds more (a close-up reaction shot runs 1.5 to 3 seconds in all). A "
    "reaction cut short kills the joke: say 'hold longer'. When a hold feels just long "
    "enough, a third of a second more is usually right.\n"
    "- A look or pose needs at least a quarter of a second of stillness to register at all.\n"
    "- Wide gag shots run 3 to 7 seconds; close-ups read faster and can be shorter. A whole "
    "short averages about 4 to 7 seconds a shot, never all the same length.\n"
    "- Gags in threes get a little quicker each time, and a beat of calm before the last, "
    "biggest gag makes it land. The last quarter of the cartoon should be about twice as "
    "tightly packed as the opening. After the biggest gag, one longer, calm closing shot.\n"
    "- Cut when the thought is finished (often on a blink), and during movement rather than "
    "after everything has stopped.")

# for the sound machine's plan, on top of "a sound for everything"
SOUND_PLAN = (
    "CARTOON, WALL TO WALL: this is a wordless slapstick cartoon with no narration, so the "
    "sound effects tell the story. Give a sound to EVERY visible action, however small: each "
    "kind of footstep (tiptoe, scamper, skid), every bonk, bounce, zip, whoosh, splat, pop, "
    "stretch, blink and double take, plus things falling, landing and wobbling. Use the "
    "classic vocabulary rather than realistic sounds wherever it fits: a fall = slide whistle "
    "down, a rise = slide whistle up; a bonk on the head = hollow bonk or anvil clang with a "
    "wobbly cymbal ring after; a slam = dull drum thud; a big crash = clattering pots and "
    "pans; a sudden stop = tyre screech; a dash = whizz or ricochet zip; a bounce or poke = "
    "boing, pitched differently each time; tiptoeing = plucked strings or xylophone steps; a "
    "blink or idea = one xylophone note or ding; stuck = hollow thunk then a pop; dizzy = "
    "whirr and tweeting birds; stretching = rising squeak. Sounds are bigger and sillier "
    "than life, dry and close, and loud enough to sit on top of the music. Mark them synced "
    "and time each to the exact moment of the action, never after it. Leave a short gap of "
    "silence just before the biggest hit of each gag: plan nothing in the half second before "
    "it. Keep ambience light. The characters' own voice sounds (gasps, grunts, giggles, sighs) "
    "are planned separately, each as its own sound of kind \"voice\": never fold one into "
    "another sound's description.")

# the characters' own wordless sounds: what they feel, think and mean to do
VOICES = (
    "CHARACTER VOICES. Nobody speaks in this cartoon, so each character's own little voice "
    "sounds are how the viewer knows what he FEELS, what he is THINKING and what he MEANS TO "
    "DO. The classic shorts did this all the time: the sniff, the gasp, the worried gulp, the "
    "pleased chuckle. Plan one for every change of feeling you can see on the face or body, "
    "at the moment it changes:\n"
    "- finding something wonderful: a sharp little gasp, then a breathless, delighted "
    "\"ooooh\" that rises, or fast excited chittering, as if he can't believe his luck\n"
    "- wanting it, claiming it: a greedy little giggle, a happy \"mine!\" squeal with no words\n"
    "- an idea: a bright, short \"ah-HA\" squeak\n"
    "- thinking, puzzled: a low curious \"hmmm?\" that rises at the end\n"
    "- making up his mind: one firm little huff or \"hmph!\" just BEFORE he acts, so the "
    "viewer knows what is coming\n"
    "- effort: straining grunts in time with each push or pull, each higher and tighter than "
    "the last, ending in a gasp for air\n"
    "- it didn't work: a grumbling mutter, a huffy snort, a long sagging \"awww\"\n"
    "- alarm: a high yelp or squeak; worry: a small whimper or a gulp\n"
    "- tired or relieved: panting, then one long sigh\n"
    "- proud of himself: a smug little \"heh-heh\" chuckle\n"
    "- content: a happy hum, a pleased munching \"mmm\"\n"
    "Rules: never words, never talking. One feeling per sound, short (about half a second to "
    "two and a half seconds). The sound starts on the still where the face or body changes, "
    "which is just AFTER the thing that causes it; effort sounds go with the movement itself. "
    "Usually one to three in a shot; a shot where nothing is felt needs none, and a held look "
    "can stay quiet. Never two voice sounds at once. Every sound_prompt begins with that "
    "character's voice, in the same words every time, so he always sounds like himself, and "
    "then says exactly what is heard and the feeling in it.")

# for checking the pictures and clips made for a cartoon
CLIPS = (
    "This film is a wordless slapstick cartoon, so also judge: can you read what the "
    "character is thinking? Is the picture the pose just BEFORE the action (so the clip has "
    "the action to do)? In the clip, is there a wind-up, then a quick action, then a moment "
    "of stillness for the reaction? A gag that floats like slow motion needs a speed of 130 "
    "to 200; a reaction that is cut off, or a mouth that moves as if talking, needs making "
    "again. Is the character on the same side of the picture, facing the same way, as the "
    "shot asks?")

# plain words for the person making the cartoon (shown in the app's guide)
LESSONS = [
    ("One small wish, one small nuisance",
     "The whole cartoon can be one idea: he wants a nap, a bee won't let him. The classic "
     "shorts filled a minute and a half with one dog and one sheet of sticky paper."),
    ("He thinks before he acts",
     "The pause where he stops, frowns and gets an idea is what makes people love him. Never "
     "trim it out: it should last about half a second to a second."),
    ("Set it up, then surprise",
     "A gag works when we expect one thing and get another. The best ones add a 'topper': "
     "the thing we expected happens, and right after it a second small surprise."),
    ("The gag is wide, the reaction is close",
     "Show the tumble in one wide shot with the camera still and all of him in the picture. "
     "Don't cut in the middle. Then cut to a close-up of his face."),
    ("Slow wind-up, fast gag, long look",
     "He gets ready slowly (so we see it coming), the gag is over in a blink (speed the clip "
     "up if it floats), and then his face holds still for a second or two. Most beginners cut "
     "the reaction too soon."),
    ("Embarrassed, never hurt",
     "His own plan backfires; nothing is mean. He's fine a second later. That's what keeps it "
     "cute."),
    ("Each try is bigger",
     "A try, a bigger try, the biggest try backfires. Start slow and make the last part of "
     "the cartoon about twice as busy as the beginning."),
    ("Everyone keeps their side",
     "If he's on the left facing right in one shot, keep him there. If he runs out on the "
     "right, he comes into the next shot from the left. If a shot faces the wrong way, flip "
     "it in Premiere (Horizontal Flip)."),
    ("Make the picture of the moment before",
     "The video AI starts from your picture, so draw the crouch, not the jump. If the picture "
     "already shows the splash, the clip has nothing left to do."),
    ("Every action gets a sound, bigger than life",
     "A slide whistle for a fall, a boing for a bounce, plucked strings for tiptoes. Put a "
     "tiny silence just before the biggest hit: it makes it funnier."),
    ("Let the music change",
     "A calm strolling tune, a sneaky tiptoe tune and a fast chase tune, and let the music "
     "stop dead for the big gag and bounce back a second later."),
    ("Cuts, not fades",
     "Inside a scene, cut straight from one shot to the next. Save a fade or dissolve for "
     "when time passes or the place changes. End on an iris-out or a slow fade."),
]
