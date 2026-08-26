#!/usr/bin/env bash
set -euo pipefail

backend_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(dirname "$backend_root")"
frontend_source="${OLV_WEB_SOURCE:-$workspace_root/Open-LLM-VTuber-Web}"
static_target="$backend_root/frontend"

if [[ ! -f "$frontend_source/package.json" ]]; then
  echo "Frontend source not found: $frontend_source" >&2
  exit 1
fi

if [[ ! -d "$static_target/.git" && ! -f "$static_target/.git" ]]; then
  echo "Frontend static submodule not initialized: $static_target" >&2
  exit 1
fi

if [[ "$frontend_source" == /tmp/* ]]; then
  echo "Refusing ephemeral frontend source: $frontend_source" >&2
  exit 1
fi

npm --prefix "$frontend_source" run build:web

if [[ ! -f "$frontend_source/dist/web/index.html" ]]; then
  echo "Frontend build did not create dist/web/index.html" >&2
  exit 1
fi

rsync -a --delete --exclude='.git' "$frontend_source/dist/web/" "$static_target/"
echo "Frontend bundle synchronized from $frontend_source to $static_target"
