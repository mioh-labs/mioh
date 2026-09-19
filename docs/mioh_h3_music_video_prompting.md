# mioh H3 Music Video Prompting

This guide defines the mioh upscaler-specific prompt contract for long-form
MiniMax H3 music-video generation. It supplements the general H3 prompt-writing
skill when a prompt will be used with mioh's `music-video` runner.

## Flat Timeline Entries

Write long-form MV prompts as explicit flat timeline entries when precise
control matters. Each entry describes one musical or visual interval with:

- absolute start and end times that cover the selected audio range without gaps;
- a transition intent, normally `cut` for a new composition and `continue` only
  when the next interval must inherit the previous generated state;
- concrete subject, setting, action, camera, lighting, and mood for that exact
  interval.

Prefer direct interval instructions over global story summaries. Repeated
global action prose can make H3 replay the same opening action in later
intervals.

## Long Prompt Structure

For long music videos, do not write one huge story paragraph and repeat it in
every interval. Use a two-layer structure:

1. a global continuity block that acts like a `prompt_prefix`;
2. flat timeline entries that describe only what changes or develops inside
   each interval.

The global block should contain stable facts only:

- subject identity, adult age, face, hair, wardrobe, and distinctive features;
- persistent location rules, visual style, camera language, and lighting style;
- music-use rules such as background music only, no lip-sync unless explicitly
  requested;
- continuity rules such as no duplicated subject, no reset, no replay, no
  time-lapse unless asked.

Each interval body should contain only local information:

- current composition or continuation state;
- concrete action development for this exact time range;
- camera movement, subject blocking, props, emotion, and visible text;
- the unfinished movement or clean boundary that should feed a following
  `.continue` interval.

For music videos, write each interval body as production notes with explicit
visual axes. This keeps Gemma from falling back to the same general mood image
for every interval. Prefer this order:

```text
LOCATION: the exact place for this interval, including what is not visible if
the previous interval's location must be avoided.
FRAMING: shot size, lens feel, subject placement, foreground/background, and
whether the image is wide, medium, close-up, profile, overhead, handheld, etc.
ACTION: what <Subject N> is doing now, how the lyric emotion changes, and what
physical state is reached by the end of the interval.
CAMERA: camera movement, speed, screen direction, focus behavior, and whether
the camera holds, tracks, circles, pushes in, or cuts away.
LIGHTING / COLOR: only the local lighting change when it differs from global
continuity.
CONTINUATION NOTE: only when this interval may be internally split; state what
must keep moving and what must not restart.
```

Do not rely on a single global phrase such as "Tokyo neon street" or
"cinematic mood" to carry many intervals. If several intervals share the same
city or character, vary at least three of these local axes: location, framing,
blocking, camera path, foreground objects, background geometry, light source,
or action endpoint.

## Face-Reference Identity Scope

When `face_only` references are used, the reference crops are identity controls
for the labeled subject only. They are not frame, outfit, body, pose,
background, lighting, camera, crop, mood, or composition references. They are
also not a general face-style palette. If the prompt includes other people,
write them so the model does not borrow the referenced face:

- define only supplied identities as `<Subject N>`;
- do not invent `<Subject 2>` or later labels for unreferenced performers;
- do not describe `<Picture N>` as a storyboard, first frame, keyframe,
  composition anchor, outfit source, body source, pose source, background
  source, lighting source, or camera-angle source unless the user explicitly
  asks for that;
- do not recreate the reference photo itself, even for the referenced subject;
- regenerate clothing, body blocking, hand pose, environment, framing, lighting,
  camera angle, and photo mood from the interval text;
- never describe an unreferenced performer as sharing the referenced face,
  hairline, eye shape, nose, mouth, or recognizable identity;
- avoid close-up face shots of unreferenced people unless the user explicitly
  asks for them;
- prefer silhouettes, back views, side profiles, motion-blurred crowds, distant
  bodies, hands, feet, reflections without facial detail, or clearly unrelated
  faces for non-subject performers;
- when a non-subject face must appear near camera, state that it is visually
  unrelated to the reference pictures and has different facial structure.

This is especially important for music videos with a partner, friends, dancers,
or crowds. If a female lead, friend group, crowd member, poster face, mirror
reflection, or background dancer is not explicitly mapped to a supplied
`<Subject N>`, do not give that person a hero face close-up. Keep the hero
identity on the referenced subject and use composition, hands, body language,
lighting, or distance for everyone else.

This style mirrors H3 chain plans that separate `prompt_prefix` from per-shot
prompts, but mioh stores it as plain prompt text. Keep the same discipline even
when writing a single flat-timeline prompt.

Recommended shape:

```text
GLOBAL CONTINUITY:
Stable subject identity, wardrobe, location rules, music rules, no duplicates,
and no replay/reset rules.

[0.000-5.167 cut]
LOCATION: rain-slick rooftop edge above Shibuya, no street-level crowd visible.
FRAMING: wide back shot of <Subject 1> at screen center with the skyline below.
ACTION: <Subject 1> stands still, lowers his shoulders, and begins the lyric
with restrained mouth movement; by the end he turns slightly toward profile.
CAMERA: slow push-in from behind, stable lens, no orbit.
LIGHTING / COLOR: cool city backlight with one warm sign reflection on his coat.
CONTINUATION NOTE: if split, continue the same turn and push-in without
returning to the first back-facing pose.

[5.167-10.125 continue]
LOCATION: same rooftop physical state.
FRAMING: side-profile medium close-up, skyline blurred behind him.
ACTION: continue from the existing partial turn; <Subject 1> finishes turning,
eyes lift toward the stars, and the vocal expression opens.
CAMERA: keep the push-in momentum, then settle into a close side profile.
CONTINUATION NOTE: do not repeat the back-facing rooftop opening.
```

Avoid putting full entry-body prose into every continuation. A `.continue`
entry may refer to the same identity, style, or intended next beat, but it
should not re-declare the original opening action as if it starts from frame
zero again.

## H3 Chain JSON Import

mioh can also read a compact H3 chain JSON prompt. This is for importing the
planning style of H3 context-loop workflows without copying their runtime code.
The supported fields are:

```json
{
  "prompt_prefix": "GLOBAL CONTINUITY: stable identity, style, music rules.",
  "fps": 24,
  "seam_taper_frames": 6,
  "shots": [
    {
      "id": "opening",
      "transition": "cut",
      "length": 124,
      "prompt": "The interval-specific action.",
      "anchor_mode": "head",
      "context_length": 22
    },
    {
      "id": "carry",
      "transition": "continue",
      "length": 119,
      "prompt": "Continue the unfinished movement without restarting."
    }
  ]
}
```

`prompt_prefix` becomes the global continuity block. Each `shot.prompt` becomes
one flat interval body. `length` and `frames` are interpreted as 24fps frame
counts unless `fps` is specified. `context_length` must match mioh's latent
continuation overlap of 22 pixel frames. `seam_taper_frames` controls how many
visible overlap frames are blended at assembly time; it must fit inside the
22-frame continuation pre-roll.

## Continue Intervals

For mioh, `.continue` intervals receive the preceding generated latent or frame
as the real physical state. Do not write continuation text as if the shot is
starting again.

The runner still sends the relevant flat-timeline entry text to `.continue`
intervals. That text is intentionally kept available for intent, mood, setting,
constraints, and allowed next developments; it is not treated as a restart pose
or a command to replay the entry from the beginning. The runner's continuation
directive tells H3 to treat the supplied latent or frame as the current physical
state and to continue from the current body pose, camera position, object state,
and motion.

When writing a prompt that may be split into `.continue` parts, include an
explicit continuation rule in the relevant entry:

```text
For continuation parts of this shot, do not restart the described action.
Treat the previous generated latent as the real current state.
Continue from whatever body pose, camera position, object state, and lighting
already exist. Use this shot text only for intent, mood, location, and the next
development. Do not return to the first pose or repeat the opening gesture.
```

If a specific action is prone to repetition, name it explicitly:

```text
Do not pick up the pen again, do not begin writing again, and do not return to
the first writing pose. Continue from the already generated hand, paper, gaze,
and camera position.
```

## Cut Intervals

Use `cut` when the next interval should be a new composition or a deliberate
visual reset. A cut entry may still preserve identity, wardrobe, lighting style,
or location if the prompt says so, but it should not rely on the previous
latent.

When all timeline entries are written as `cut`, remember that a long entry can
still be internally split by mioh. The later parts of that same entry may run as
continuations, so write long entries with continuation-safe language.

## Action Wording

Avoid phrasing repeated states as repeated beginnings. Prefer state and
development language:

- Weak: `She writes a letter at the desk.`
- Better: `She is at the desk with a partly written letter, continuing from the
  current hand and paper position without restarting the writing gesture.`

For props, text, paper, handwriting, phones, doors, mirrors, and seated desk
actions, explicitly say whether the action should continue, stop, or develop.

## Camera and Composition

For smooth continuation, write camera instructions as persistent constraints:

```text
Maintain the current camera height, lens feel, subject scale, screen direction,
background geometry, and lighting continuity across continuation parts.
```

When the composition must change, make that a new `cut` entry instead of asking
a `.continue` interval to both preserve and reframe.
