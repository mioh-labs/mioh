#!/bin/zsh
# Remux every .mkv in a directory to .mp4 without re-encoding.
#
# AVFoundation cannot open Matroska at all, so mioh's realtime preview has to
# remux an .mkv before it can play it. Doing that once here means the preview
# opens the file directly instead of converting it on every run.
#
# Streams are copied byte-for-byte: same video, same audio, same quality.
# Subtitles, chapters, attachments and any third stream are dropped, because
# MP4 cannot carry the Matroska forms of most of them.
#
# Usage:
#   ./remux_mkv_to_mp4.zsh [directory] [--force] [--dry-run]
#
#   directory   defaults to the current directory
#   --force     overwrite an .mp4 that already exists
#   --dry-run   print what would run and change nothing
#
# Originals are never modified or deleted.

set -euo pipefail

DIRECTORY="."
FORCE=0
DRY_RUN=0

for argument in "$@"; do
  case "$argument" in
    --force) FORCE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    -*)
      print -u2 "Unknown option: $argument"
      exit 2
      ;;
    *) DIRECTORY="$argument" ;;
  esac
done

if [[ ! -d "$DIRECTORY" ]]; then
  print -u2 "Not a directory: $DIRECTORY"
  exit 1
fi

# Prefer whatever is on PATH, then the copy mioh bundles.
FFMPEG="${FFMPEG:-$(command -v ffmpeg || true)}"
FFPROBE="${FFPROBE:-$(command -v ffprobe || true)}"
BUNDLED="/Applications/mioh.app/Contents/Resources/bin"
[[ -z "$FFMPEG" && -x "$BUNDLED/ffmpeg" ]] && FFMPEG="$BUNDLED/ffmpeg"
[[ -z "$FFPROBE" && -x "$BUNDLED/ffprobe" ]] && FFPROBE="$BUNDLED/ffprobe"
if [[ -z "$FFMPEG" || -z "$FFPROBE" ]]; then
  print -u2 "ffmpeg/ffprobe not found. Install them (brew install ffmpeg) or set FFMPEG/FFPROBE."
  exit 1
fi

typeset -a SOURCES
SOURCES=("$DIRECTORY"/*.mkv(N))
if (( ${#SOURCES} == 0 )); then
  print "No .mkv files in $DIRECTORY"
  exit 0
fi

converted=0
skipped=0
failed=0

for source in "${SOURCES[@]}"; do
  target="${source:r}.mp4"
  name="${source:t}"

  if [[ -e "$target" && "$FORCE" != 1 ]]; then
    print "skip     $name  (${target:t} exists; --force to replace)"
    (( skipped += 1 ))
    continue
  fi

  codec="$("$FFPROBE" -v error -select_streams v:0 \
    -show_entries stream=codec_name -of csv=p=0 "$source" 2>/dev/null || true)"
  if [[ -z "$codec" ]]; then
    print -u2 "FAIL     $name  (no video stream)"
    (( failed += 1 ))
    continue
  fi

  # An HEVC track remuxed into MP4 keeps its Matroska sample entry, which
  # AVFoundation rejects. Retagging it as hvc1 costs nothing and is what makes
  # the result openable; other codecs keep whatever tag ffmpeg picks.
  typeset -a tag
  tag=()
  case "$codec" in
    hevc|h265) tag=(-tag:v hvc1) ;;
  esac

  typeset -a command
  command=(
    "$FFMPEG" -hide_banner -loglevel error -stats
    -i "$source"
    -map 0:v:0 -map "0:a?"
    -c copy "${tag[@]}"
    -sn -dn -map_chapters -1
    -movflags +faststart
    -f mp4
    -y "$target.partial"
  )

  if (( DRY_RUN )); then
    print "would remux $name  ($codec)"
    print "  ${(j: :)${(q)command}}"
    continue
  fi

  print "remux    $name  ($codec)"
  if ! "${command[@]}"; then
    rm -f "$target.partial"
    print -u2 "FAIL     $name  (ffmpeg error)"
    (( failed += 1 ))
    continue
  fi

  # Only publish the .mp4 once it reads back as a real video file, so an
  # interrupted run never leaves a broken sibling next to the original.
  if ! "$FFPROBE" -v error -select_streams v:0 \
    -show_entries stream=codec_name -of csv=p=0 "$target.partial" >/dev/null
  then
    rm -f "$target.partial"
    print -u2 "FAIL     $name  (output did not verify)"
    (( failed += 1 ))
    continue
  fi

  mv -f "$target.partial" "$target"
  print "ok       ${target:t}"
  (( converted += 1 ))
done

print ""
print "converted $converted, skipped $skipped, failed $failed"
print "Originals were left untouched in $DIRECTORY"
(( failed == 0 ))
