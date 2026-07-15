#!/bin/sh
set -eu

pinned_runtime_revision='bf2c86ddc0685f580595954056c2e77ebabfab4f'

if [ "$#" -ne 6 ]; then
  printf '%s\n' 'usage: build_runtime_bundle.sh darwin|linux arm64|amd64 /path/to/llama-server expected-sha256 /path/to/source-checkout /path/to/runtime.tar.gz' >&2
  exit 2
fi

target_os="$1"
target_arch="$2"
server="$3"
expected_sha256="$4"
source_checkout="$5"
output="$6"

case "$target_os" in
  darwin|linux) ;;
  *) printf '%s\n' "unsupported runtime operating system: $target_os" >&2; exit 2 ;;
esac
case "$target_arch" in
  arm64|amd64) ;;
  *) printf '%s\n' "unsupported runtime architecture: $target_arch" >&2; exit 2 ;;
esac
if [ ! -f "$server" ] || [ ! -x "$server" ] || [ -L "$server" ]; then
  printf '%s\n' 'llama-server must be a regular executable, not a symlink' >&2
  exit 2
fi
case "$expected_sha256" in
  *[!0-9a-f]*|'') printf '%s\n' 'expected server SHA-256 must be 64 lowercase hex characters' >&2; exit 2 ;;
esac
if [ "${#expected_sha256}" -ne 64 ]; then
  printf '%s\n' 'expected server SHA-256 must be 64 lowercase hex characters' >&2
  exit 2
fi
if [ ! -d "$source_checkout" ] || [ -L "$source_checkout" ]; then
  printf '%s\n' 'runtime source checkout is missing or unsafe' >&2
  exit 2
fi
command -v git >/dev/null 2>&1 || { printf '%s\n' 'git is required to verify runtime provenance' >&2; exit 2; }
source_checkout=$(CDPATH= cd -- "$source_checkout" && pwd -P)
server_directory=$(CDPATH= cd -- "$(dirname -- "$server")" && pwd -P)
server="$server_directory/$(basename -- "$server")"
case "$server" in
  "$source_checkout"/*) ;;
  *) printf '%s\n' 'runtime server must be built inside the pinned source checkout' >&2; exit 2 ;;
esac
server_revision=$(git -C "$source_checkout" rev-parse HEAD 2>/dev/null || true)
if [ "$server_revision" != "$pinned_runtime_revision" ]; then
  printf '%s\n' 'server revision does not match the pinned runtime revision' >&2
  exit 2
fi

if command -v shasum >/dev/null 2>&1; then
  actual_sha256=$(shasum -a 256 "$server" | awk '{print $1}')
elif command -v sha256sum >/dev/null 2>&1; then
  actual_sha256=$(sha256sum "$server" | awk '{print $1}')
else
  printf '%s\n' 'a SHA-256 utility is required to package the runtime' >&2
  exit 2
fi
if [ "$actual_sha256" != "$expected_sha256" ]; then
  printf '%s\n' 'server SHA-256 does not match the release input' >&2
  exit 2
fi

dependencies=$(mktemp "${TMPDIR:-/tmp}/incubus-dependencies.XXXXXX")
trap 'rm -f "$dependencies"' EXIT HUP INT TERM
case "$target_os" in
  darwin)
    command -v otool >/dev/null 2>&1 || { printf '%s\n' 'otool is required to inspect runtime dependencies' >&2; exit 2; }
    otool -L "$server" > "$dependencies"
    ;;
  linux)
    command -v readelf >/dev/null 2>&1 || { printf '%s\n' 'readelf is required to inspect runtime dependencies' >&2; exit 2; }
    readelf -d "$server" > "$dependencies"
    ;;
esac
if grep -Eiq 'lib(llama|ggml)' "$dependencies"; then
  printf '%s\n' 'runtime server has unbundled dynamic inference-library dependencies' >&2
  exit 2
fi
"$server" --version >/dev/null

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
output_dir=$(dirname -- "$output")
mkdir -p "$output_dir"
output=$(CDPATH= cd -- "$output_dir" && pwd)/$(basename -- "$output")
staging=$(mktemp -d "${TMPDIR:-/tmp}/incubus-runtime.XXXXXX")
trap 'rm -rf "$staging" "$dependencies"' EXIT HUP INT TERM
mkdir -p "$staging/bin" "$staging/metadata"

(
  cd "$script_dir"
  CGO_ENABLED=0 GOOS="$target_os" GOARCH="$target_arch" \
    go build -buildvcs=false -trimpath -ldflags '-s -w' -o "$staging/bin/incubus-runtime" ./cmd/incubus-runtime
)
cp "$server" "$staging/bin/llama-server"
chmod 700 "$staging/bin/incubus-runtime" "$staging/bin/llama-server"
printf '%s\n' "$pinned_runtime_revision" > "$staging/metadata/runtime-revision"
chmod 600 "$staging/metadata/runtime-revision"
"$staging/bin/llama-server" --version >/dev/null

temporary="$output.tmp.$$"
rm -f "$temporary"
tar -czf "$temporary" -C "$staging" bin metadata
mv "$temporary" "$output"
