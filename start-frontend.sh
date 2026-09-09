#!/usr/bin/env bash
#
# Start the results dashboard (dashboard/, Next.js).
#
#   ./start-frontend.sh                 dev server with hot reload, port 3000
#   ./start-frontend.sh --port 4000     dev server on another port
#   ./start-frontend.sh --prod          production build, then serve it
#   ./start-frontend.sh --build         static export to dashboard/out/ and exit
#
# Dependencies install themselves on first run. Node 20 or newer is required:
# Next 16 will not start on older runtimes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$REPO_ROOT/dashboard"
PORT=3000
MODE="dev"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="${2:?--port needs a number}"; shift 2 ;;
    --prod) MODE="prod"; shift ;;
    --build) MODE="build"; shift ;;
    -h|--help) sed -n '3,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

if [[ ! -d "$APP_DIR" ]]; then
  echo "error: $APP_DIR is missing. Is this the right repository?" >&2
  exit 1
fi

if ! command -v node >/dev/null 2>&1; then
  echo "error: node is not installed. Next 16 needs Node 20 or newer." >&2
  echo "  macOS:  brew install node" >&2
  exit 1
fi

NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]')"
if (( NODE_MAJOR < 20 )); then
  echo "error: Node $(node -v) is too old. Next 16 needs Node 20 or newer." >&2
  exit 1
fi

cd "$APP_DIR"

# Reinstall when the lockfile is newer than the installed tree, so a pulled dependency
# change does not surface later as a confusing missing-module error.
if [[ ! -d node_modules || package-lock.json -nt node_modules ]]; then
  echo "==> installing dependencies (first run takes a minute)"
  npm install --no-audit --no-fund
fi

if lsof -i :"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "error: port $PORT is already in use." >&2
  echo "  Free it, or start on another port:  ./start-frontend.sh --port $((PORT + 1))" >&2
  exit 1
fi

case "$MODE" in
  build)
    echo "==> building static export"
    npm run build
    echo
    echo "Static export written to dashboard/out/"
    echo "Every figure is compiled in from lib/data.ts. Nothing is fetched at runtime."
    ;;
  prod)
    echo "==> production build"
    npm run build
    echo
    echo "==> serving the static export on http://localhost:$PORT"
    echo "    Ctrl-C to stop."
    npx --yes serve out -l "$PORT"
    ;;
  dev)
    echo "==> dashboard on http://localhost:$PORT"
    echo "    Edit dashboard/lib/data.ts to change any figure on the page."
    echo "    Ctrl-C to stop."
    echo
    npm run dev -- --port "$PORT"
    ;;
esac
