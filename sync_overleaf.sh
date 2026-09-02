#!/usr/bin/env bash
set -euo pipefail

VIDEO_RLM=/dataheart/hussainahmad/video-rlm
SOURCE="$VIDEO_RLM/paper"
SOURCE_TEX="$SOURCE/priority_starts_before_gpu.tex"
OVERLEAF=/dataheart/hussainahmad/overleaf-video-rlm-paper

if [ ! -d "$OVERLEAF/.git" ]; then
    echo "Overleaf clone not found: $OVERLEAF"
    exit 1
fi

if [ ! -f "$SOURCE_TEX" ]; then
    echo "Paper source not found: $SOURCE_TEX"
    exit 1
fi

OVERLEAF_BRANCH=$(git -C "$OVERLEAF" branch --show-current)

if [ -z "$OVERLEAF_BRANCH" ]; then
    echo "Could not determine the checked-out Overleaf branch."
    exit 1
fi

echo "Pulling any changes made in Overleaf..."
git -C "$OVERLEAF" pull --rebase origin "$OVERLEAF_BRANCH"

echo "Copying the local manuscript and figures into the Overleaf checkout..."
cp "$SOURCE_TEX" "$OVERLEAF/main.tex"
mkdir -p "$OVERLEAF/figures"
rsync -av \
    --exclude='*.aux' \
    --exclude='*.log' \
    --exclude='*.out' \
    --exclude='*.fls' \
    --exclude='*.fdb_latexmk' \
    --exclude='*.synctex.gz' \
    "$SOURCE/figures/" \
    "$OVERLEAF/figures/"

git -C "$OVERLEAF" add -A

if git -C "$OVERLEAF" diff --cached --quiet; then
    echo "Overleaf is already up to date."
    exit 0
fi

git -C "$OVERLEAF" commit \
    -m "Synchronize paper from video-rlm"

git -C "$OVERLEAF" push origin "$OVERLEAF_BRANCH"

echo "Overleaf project updated."
