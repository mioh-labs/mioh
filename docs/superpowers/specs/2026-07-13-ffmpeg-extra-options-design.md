# mioh Additional FFmpeg Options Design

## Goal

Allow users to add detailed final-output encoder options from a free-form text box in mioh. The options augment the selected automatic, preset, or custom encoding configuration. When an option is specified both by the base configuration and the text box, the text-box value wins.

## Scope

The feature applies only to the encoder used to write the restored output video. It does not inject arbitrary arguments into the FFmpeg commands used for segment splitting or merging, and it does not allow replacement of input or output paths.

The parallel-processing CLI receives the same merge behavior through its existing `--encoder-options` argument. The underlying `lada-cli` interface remains unchanged.

## User Interface

The Output tab displays an always-visible multiline field labeled `追加FFmpegオプション`. It is available in automatic, preset, and custom encoding modes.

Example input:

```text
-pix_fmt yuv420p10le -profile:v main10 -b:v 20M -maxrate 25M
```

The existing custom-mode encoder field remains responsible for choosing the encoder. The new field is responsible only for encoder options.

## Option Resolution

The parallel processor resolves the final encoder and options in this order:

1. Select the base encoder and base options from automatic optimization, the selected preset, or the custom encoder configuration.
2. Parse the additional options with `shlex` so quoted values remain intact.
3. Merge additional options into the base options.
4. Replace the value of an existing option when the additional input uses the same option key.
5. Preserve base options that are not overridden and append new option keys in input order.
6. Pass the resolved encoder and one final option string to `lada-cli`.

An option key is a token beginning with `-` followed by a non-numeric character. This distinguishes keys from negative numeric values such as `-1`. The following token is its value unless that token is another option key. Valueless flags are supported. If a key occurs more than once within the same input, the last occurrence wins. This intentionally matches the option model already consumed by Lada's PyAV output writer; it is not a general-purpose FFmpeg command-line parser.

## Validation and Errors

Before processing begins, the additional option string is parsed using the same helper used for the merge. Unbalanced quotes and non-empty tokens appearing before the first option key are rejected with a concise error. An empty field is valid and preserves existing behavior.

Validation failures are shown in mioh's status and log without launching restoration. CLI validation failures exit with a nonzero status and identify the invalid option string.

## Logging

The processor emits one concise line containing the resolved encoder and final encoder options. It does not print duplicate per-frame or per-segment copies of the configuration.

## Compatibility

- Existing presets keep their current output when the additional field is empty.
- Existing custom encoder behavior remains available.
- Existing automatic optimization remains the base configuration and can be overridden by additional options.
- `--encoder-options` used with `--encoding-preset` now augments the preset instead of being ignored.
- The standalone bundle continues using its bundled FFmpeg and ffprobe binaries.

## Testing

Automated tests cover:

- empty additional options preserving the base options;
- new options being appended;
- duplicate keys being overridden by the additional value;
- quoted values and valueless flags;
- malformed input being rejected;
- preset, automatic, and custom modes using the same merge path;
- the Swift UI exposing the multiline field and forwarding its value;
- no option injection into split or merge FFmpeg command construction.

A standalone app build verifies that the field is present and that the generated processing arguments contain the expected `--encoder-options` value.
