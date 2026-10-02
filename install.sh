#!/bin/sh
#
# Slipstream installer for Apple Silicon Macs.
#
#   curl -fsSL https://github.com/mzinner/slipstream/raw/main/install.sh | sh
#
# Picks the package for this Mac from a GitHub release, verifies it against the
# release's SHA256SUMS, unpacks it into ~/.local/share/slipstream/<version> and
# links `slipstream` into ~/.local/bin. The two newest versions are kept.
#
# Environment:
#   SLIPSTREAM_TAG     install this release tag instead of the newest, e.g. v26.10.0
#   SLIPSTREAM_PREFIX  where versions are unpacked  (default ~/.local/share/slipstream)
#   SLIPSTREAM_BINDIR  where the command is linked  (default ~/.local/bin)
#   SLIPSTREAM_REPO    owner/repo to install from   (default mzinner/slipstream)
#   SLIPSTREAM_TOKEN   GitHub token for a private repository (GH_TOKEN, GITHUB_TOKEN
#                      and `gh auth token` are tried too)
#
# POSIX sh, so it runs from a pipe into whatever /bin/sh is.
set -eu

REPO="${SLIPSTREAM_REPO:-mzinner/slipstream}"
PREFIX="${SLIPSTREAM_PREFIX:-$HOME/.local/share/slipstream}"
BINDIR="${SLIPSTREAM_BINDIR:-$HOME/.local/bin}"
TOKEN="${SLIPSTREAM_TOKEN:-${GH_TOKEN:-${GITHUB_TOKEN:-}}}"
ARCH_TOKEN="arm-64bit"

die() { printf 'install.sh: %s\n' "$*" >&2; exit 1; }
info() { printf '==> %s\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }

case "${1:-}" in
  -h|--help)
    sed -n '2,/^set -eu/p' "$0" 2>/dev/null | sed '$d; s/^# \{0,1\}//'
    exit 0 ;;
  "") ;;
  *) die "unknown option: $1 (try --help)" ;;
esac

[ "$(uname -s)" = Darwin ] || die "Slipstream runs on macOS only (this is $(uname -s))"
case "$(uname -m)" in
  arm64|aarch64) ;;
  *) die "Slipstream needs an Apple Silicon Mac (this CPU is $(uname -m))" ;;
esac
need curl
need ditto
need shasum
need awk

# The package is built for a macOS major version; it runs on that version and later.
MACOS=$(sw_vers -productVersion)
MACOS_MAJOR=${MACOS%%.*}
info "Detected macOS $MACOS on Apple Silicon"

# Without a tag, /releases/latest/download/ always points at the newest release,
# so this script never needs to know a version.
if [ -n "${SLIPSTREAM_TAG:-}" ]; then
  BASE="https://github.com/$REPO/releases/download/$SLIPSTREAM_TAG"
  WHICH="$SLIPSTREAM_TAG"
else
  BASE="https://github.com/$REPO/releases/latest/download"
  WHICH="latest"
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT INT TERM

# Release assets of a private repository are only reachable through the API, by id.
USE_API=0
api_assets() {
  if [ -n "${SLIPSTREAM_TAG:-}" ]; then
    url="https://api.github.com/repos/$REPO/releases/tags/$SLIPSTREAM_TAG"
  else
    url="https://api.github.com/repos/$REPO/releases/latest"
  fi
  curl -fsSL --retry 3 -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json" \
    -o "$TMP/release.json" "$url" || return 1
  # tag on the first line, then "id name" per asset; python3 ships with the
  # command line tools, plutil/jq would also do.
  /usr/bin/python3 -c 'import json, sys
release = json.load(open(sys.argv[1]))
print(release["tag_name"])
for asset in release["assets"]:
    print(asset["id"], asset["name"])' "$TMP/release.json" > "$TMP/release.txt" || return 1
  WHICH=$(head -n 1 "$TMP/release.txt")
  USE_API=1
}

fetch() {  # fetch <asset name> <file> [curl options]
  if [ "$USE_API" = 0 ]; then
    curl -fL --retry 3 ${3:-} -o "$2" "$BASE/$1"
    return
  fi
  id=$(awk -v name="$1" 'NR > 1 && $2 == name { print $1; exit }' "$TMP/release.txt")
  [ -n "${id:-}" ] || { echo "release $WHICH has no asset $1" >&2; return 1; }
  curl -fL --retry 3 ${3:-} -H "Authorization: Bearer $TOKEN" -H "Accept: application/octet-stream" \
    -o "$2" "https://api.github.com/repos/$REPO/releases/assets/$id"
}

if [ -n "$TOKEN" ]; then
  api_assets || die "could not read the $WHICH release of $REPO with the given token"
fi

# SHA256SUMS lists every package with its checksum: it is the release's index.
info "Fetching the package list of the $WHICH release of $REPO"
if ! fetch SHA256SUMS "$TMP/SHA256SUMS" -sS 2>"$TMP/fetch.err"; then
  # A private repository answers 404 just like a missing release; try gh's login once.
  if [ "$USE_API" = 0 ] && command -v gh >/dev/null 2>&1; then
    TOKEN=$(gh auth token 2>/dev/null || true)
    [ -z "$TOKEN" ] || { api_assets && fetch SHA256SUMS "$TMP/SHA256SUMS" -sS; } || true
  fi
  if [ ! -s "$TMP/SHA256SUMS" ]; then
    sed 's/^/  /' "$TMP/fetch.err" >&2 || true
    die "could not download SHA256SUMS from the $WHICH release of $REPO.
  Does the release exist? For a private repository, run 'gh auth login' or set SLIPSTREAM_TOKEN."
  fi
fi

# Choose the package built for the newest macOS major version not above this one.
BEST=""
BEST_MAJOR=0
while read -r _sum name; do
  case "$name" in
    slipstream-*-macos*-"$ARCH_TOKEN".zip) ;;
    *) continue ;;
  esac
  major=$(printf '%s' "$name" | sed -n "s/.*-macos\([0-9]*\)-$ARCH_TOKEN\.zip$/\1/p")
  [ -n "$major" ] || continue
  [ "$major" -le "$MACOS_MAJOR" ] || continue
  if [ "$major" -gt "$BEST_MAJOR" ]; then BEST="$name"; BEST_MAJOR="$major"; fi
done < "$TMP/SHA256SUMS"
[ -n "$BEST" ] || {
  echo "install.sh: no package in the $WHICH release fits macOS $MACOS. It has:" >&2
  awk '{ print "  " $2 }' "$TMP/SHA256SUMS" >&2
  exit 1
}
info "Selected $BEST"

info "Downloading"
fetch "$BEST" "$TMP/$BEST" --progress-bar || die "could not download $BEST"

info "Verifying checksum"
EXPECTED=$(awk -v name="$BEST" '$2 == name { print $1 }' "$TMP/SHA256SUMS")
ACTUAL=$(shasum -a 256 "$TMP/$BEST" | awk '{ print $1 }')
[ "$ACTUAL" = "$EXPECTED" ] || die "checksum mismatch for $BEST
  expected: $EXPECTED
  actual:   $ACTUAL"

# The zip holds one slipstream-<version>-macos<N>-arm-64bit folder. It is renamed
# to just the version: versions sit side by side under PREFIX.
info "Unpacking into $PREFIX"
mkdir -p "$TMP/unpack" "$PREFIX"
ditto -x -k "$TMP/$BEST" "$TMP/unpack"
TOPDIR=$(ls "$TMP/unpack")
[ -d "$TMP/unpack/$TOPDIR" ] || die "unexpected layout in $BEST"
VERSION=$(printf '%s' "$TOPDIR" | sed -n 's/^slipstream-\([0-9][0-9.]*\)-.*/\1/p')
[ -n "$VERSION" ] || VERSION="$TOPDIR"
rm -rf "${PREFIX:?}/$VERSION"
mv "$TMP/unpack/$TOPDIR" "$PREFIX/$VERSION"

BIN="$PREFIX/$VERSION/bin/slipstream"
[ -x "$BIN" ] || die "no executable at $BIN after unpacking"
mkdir -p "$BINDIR"
ln -sfn "$BIN" "$BINDIR/slipstream"
info "Installed $("$BIN" --version 2>/dev/null || echo "$VERSION")"
info "Command: $BINDIR/slipstream -> $BIN"

# Keep the version just installed and the newest of the others: one way back.
# Only names made of digits and dots count as versions; nothing else is touched.
is_version() {
  [ -d "$1" ] && [ ! -L "$1" ] || return 1
  case "${1##*/}" in [0-9]*) ;; *) return 1 ;; esac
  case "${1##*/}" in *[!0-9.]*) return 1 ;; esac
}
newer() {  # is dotted version $1 greater than $2?
  awk -v a="$1" -v b="$2" 'BEGIN { na = split(a, x, "."); nb = split(b, y, ".")
    for (i = 1; i <= (na > nb ? na : nb); i++) { if (x[i] + 0 != y[i] + 0) exit !(x[i] + 0 > y[i] + 0) }
    exit 1 }'
}
KEEP=""
for dir in "$PREFIX"/*; do
  is_version "$dir" || continue
  name=${dir##*/}
  [ "$name" != "$VERSION" ] || continue
  if [ -z "$KEEP" ] || newer "$name" "$KEEP"; then KEEP="$name"; fi
done
for dir in "$PREFIX"/*; do
  is_version "$dir" || continue
  name=${dir##*/}
  [ "$name" != "$VERSION" ] && [ "$name" != "$KEEP" ] || continue
  info "Removing superseded version $name"
  rm -rf "${PREFIX:?}/$name"
done
[ -z "$KEEP" ] || info "Kept previous version $KEEP"

case ":$PATH:" in
  *":$BINDIR:"*) ;;
  *) printf '\n%s is not on your PATH. Add it with:\n\n    export PATH="%s:$PATH"\n\n' "$BINDIR" "$BINDIR" ;;
esac
printf '\nServe a model:\n\n    slipstream serve --model <owner/repo or a local model folder>\n'
