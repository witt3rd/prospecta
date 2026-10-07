#!/usr/bin/env bash
# Rebuild the Prospecta explainer (PDF + PPTX) with the house explainer builder, pinned to one spire-venue commit.
#   ./build.sh [OUT]        OUT defaults to ./out; the published files land beside this script
# Needs: git, node, uv, pdfinfo and pdftotext (poppler), tesseract, ImageMagick (magick), and Playwright with its Chromium
#   (set PLAYWRIGHT_MODULE to the playwright module directory if render.mjs cannot find it). The venue repository
#   (janus-infra/spire-venue) must be readable: override its address with SPIRE_VENUE_REPO.
set -euo pipefail
PIN=a69d1d79425583525cef616cf98afc3252b74453          # spire-venue commit of the builder files (see README.md)
REPO="${SPIRE_VENUE_REPO:-github-janus:janus-infra/spire-venue.git}"
SHA_THEME=5dab3737e9734760e97e458cd30d662a73dc7c4ed43772e23f74933bf4e8b5ec
SHA_RENDER=6549fef40f2c15969410fd24443927ec5f8e1ab2a09f19addb6e2d1c75e27880
SHA_FIG=148c971f733ec444feb798bcd8672f5b667969498e972c94744a3aaed3f54e87
here="$(cd "$(dirname "$0")" && pwd)"; out="${1:-$here/out}"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
git clone --quiet --no-checkout "$REPO" "$tmp/venue"
git -C "$tmp/venue" checkout --quiet "$PIN" -- docs/explainers/theme.css docs/explainers/render.mjs docs/explainers/spire/figures.py
cp "$tmp/venue/docs/explainers/theme.css" "$tmp/venue/docs/explainers/render.mjs" "$here/"
head -n 116 "$tmp/venue/docs/explainers/spire/figures.py" > "$here/fig.py"   # the figure primitives, verbatim
for pair in "theme.css $SHA_THEME" "render.mjs $SHA_RENDER" "fig.py $SHA_FIG"; do
  set -- $pair; echo "$2  $here/$1" | sha256sum --check --quiet || { echo "$1 differs from the pinned builder file" >&2; exit 1; }
done
if [ -z "${PLAYWRIGHT_MODULE:-}" ]; then
  for c in "$HOME"/.local/share/mise/installs/npm-playwright/latest/node_modules/playwright; do [ -d "$c" ] && export PLAYWRIGHT_MODULE="$c"; done
fi
cd "$here"
uv run --quiet --with python-pptx==1.0.2 --with pillow python3 build.py --out "$out" --publish
