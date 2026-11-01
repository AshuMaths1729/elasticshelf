#!/usr/bin/env bash
#
# ElasticShelf one-command launcher.
#
# Usage:
#   ./start.sh              # setup venv + install deps + seed data + start server + open browser
#   source start.sh         # same, and keeps .venv activated in your shell after Ctrl-C
#
# What it does:
#   1. Creates elasticshelf/.venv if missing
#   2. Activates the virtualenv and installs elasticshelf/requirements.txt (cached via SHA-256)
#   3. Copies elasticshelf/.env.example -> elasticshelf/.env if missing
#   4. Seeds catalog/weather data if missing (offline-safe)
#   5. Starts uvicorn on http://127.0.0.1:8000 and launches it in your browser
#
# Flags:
#   --port PORT       Port to bind (default: 8000, or $PORT)
#   --host HOST       Host to bind (default: 127.0.0.1, or $HOST)
#   --no-browser      Do not open the browser automatically
#   --seed            Force re-seeding catalog and weather data
#   --reinstall       Force reinstalling pip dependencies
#   --skip-install    Skip pip dependency check/install
#   -h, --help        Show this help message
#

# --- Detect script path & sourced vs executed (works in bash and zsh) --------
if [[ -n "${BASH_VERSION:-}" ]]; then
    _SCRIPT_PATH="${BASH_SOURCE[0]}"
    [[ "${BASH_SOURCE[0]}" != "${0}" ]] && _SOURCED=1 || _SOURCED=0
elif [[ -n "${ZSH_VERSION:-}" ]]; then
    _SCRIPT_PATH="${(%):-%x}"
    [[ "${ZSH_EVAL_CONTEXT:-}" == *:file* ]] && _SOURCED=1 || _SOURCED=0
else
    _SCRIPT_PATH="$0"
    _SOURCED=0
fi

if [[ "$_SOURCED" == "0" ]]; then
    set -euo pipefail
fi

_exit() { # _exit <code> [message]
    local code="$1"; shift
    if [[ $# -gt 0 ]]; then
        if [[ "$code" == "0" ]]; then
            echo "$*"
        else
            echo "error: $*" >&2
        fi
    fi
    if [[ "$_SOURCED" == "1" ]]; then
        return "$code" 2>/dev/null || exit "$code"
    else
        exit "$code"
    fi
}

# --- Parse arguments ---------------------------------------------------------
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
OPEN_BROWSER=1
FORCE_SEED=0
FORCE_INSTALL=0
SKIP_INSTALL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            [[ $# -ge 2 ]] || { _exit 1 "--port requires a value"; return 1; }
            PORT="$2"; shift 2 ;;
        --host)
            [[ $# -ge 2 ]] || { _exit 1 "--host requires a value"; return 1; }
            HOST="$2"; shift 2 ;;
        --no-browser)   OPEN_BROWSER=0; shift ;;
        --seed)         FORCE_SEED=1; shift ;;
        --reinstall)    FORCE_INSTALL=1; shift ;;
        --skip-install) SKIP_INSTALL=1; shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$_SCRIPT_PATH" | sed 's/^# \{0,1\}//'
            _exit 0; return 0 ;;
        *)
            _exit 1 "unknown flag: $1 (try --help)"; return 1 ;;
    esac
done

# --- Resolve paths -----------------------------------------------------------
ROOT="$(cd "$(dirname "$_SCRIPT_PATH")" && pwd)"
if [[ -d "$ROOT/elasticshelf/backend" ]]; then
    APP="$ROOT/elasticshelf"
elif [[ -d "$ROOT/backend" ]]; then
    APP="$ROOT"
else
    _exit 1 "could not locate elasticshelf/backend directory from $ROOT"; return 1
fi

VENV="$APP/.venv"
PY="$VENV/bin/python"
URL="http://${HOST}:${PORT}"

command -v python3 >/dev/null 2>&1 || { _exit 1 "python3 not found — install Python 3.9+ first."; return 1; }

# --- 1. Create virtualenv if missing -----------------------------------------
if [[ ! -x "$PY" ]]; then
    echo "==> [1/5] Creating virtual environment at ${VENV#"$ROOT/"}"
    python3 -m venv "$VENV" || { _exit 1 "failed to create virtualenv"; return 1; }
else
    echo "==> [1/5] Found virtual environment at ${VENV#"$ROOT/"}"
fi

# --- 2. Activate virtualenv & install requirements ---------------------------
echo "==> [2/5] Activating virtual environment"
# shellcheck disable=SC1091
source "$VENV/bin/activate" || { _exit 1 "failed to activate virtualenv"; return 1; }

if [[ "$SKIP_INSTALL" == "1" ]]; then
    echo "    Skipping dependency check (--skip-install)"
else
    STAMP="$VENV/.requirements.sha"
    CUR_HASH="$("$PY" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$APP/requirements.txt")"
    OLD_HASH="$(cat "$STAMP" 2>/dev/null || true)"
    if [[ "$FORCE_INSTALL" == "1" || "$CUR_HASH" != "$OLD_HASH" ]]; then
        echo "    Installing dependencies from requirements.txt (first run may take 1-2 min)..."
        "$PY" -m pip install --upgrade pip -q || { _exit 1 "pip upgrade failed"; return 1; }
        "$PY" -m pip install -r "$APP/requirements.txt" || { _exit 1 "pip install failed"; return 1; }
        printf '%s' "$CUR_HASH" > "$STAMP"
    else
        echo "    Dependencies up to date"
    fi
fi

# --- 3. Ensure .env configuration exists -------------------------------------
if [[ ! -f "$APP/.env" ]]; then
    echo "==> [3/5] Creating ${APP#"$ROOT/"}/.env from .env.example"
    cp "$APP/.env.example" "$APP/.env"
else
    echo "==> [3/5] Configuration ${APP#"$ROOT/"}/.env ready"
fi

# --- 4. Seed catalog & weather data if needed --------------------------------
echo "==> [4/5] Checking seed data"
seed_if_missing() { # <relpath> <module> <label>
    local rel="$1" mod="$2" label="$3"
    if [[ "$FORCE_SEED" == "1" || ! -s "$APP/$rel" ]]; then
        echo "    Seeding $label (python -m $mod)..."
        ( cd "$APP" && "$PY" -m "$mod" ) || echo "    warning: $label seeding failed — continuing with offline fallback"
    else
        echo "    $label ready ($rel)"
    fi
}
seed_if_missing "data/catalog.json" "scripts.generate_synthetic" "catalog"
seed_if_missing "data/weather.json" "scripts.fetch_open_data"   "weather"

# --- Browser helper ----------------------------------------------------------
open_browser() { # <url>
    local target="$1"
    case "$(uname -s)" in
        Darwin) open "$target" ;;
        MINGW*|MSYS*|CYGWIN*) start "" "$target" ;;
        *)
            if command -v xdg-open >/dev/null 2>&1; then
                xdg-open "$target" >/dev/null 2>&1 &
            else
                echo "    Open $target in your browser"
            fi
            ;;
    esac
}

# --- 5. Launch server & open browser -----------------------------------------
if curl -sf -o /dev/null --max-time 2 "${URL}/api/v1/sim/status" 2>/dev/null; then
    echo "==> [5/5] Server is already running at $URL"
    if [[ "$OPEN_BROWSER" == "1" ]]; then
        echo "    Opening $URL in browser..."
        open_browser "$URL" 2>/dev/null || true
    fi
    _exit 0 "Ready at $URL (virtualenv active: $VIRTUAL_ENV)"
    return 0 2>/dev/null || exit 0
fi

echo "==> [5/5] Starting ElasticShelf server at $URL"
(
    cd "$APP"
    exec "$PY" -m uvicorn backend.main:app --host "$HOST" --port "$PORT"
) &
SERVER_PID=$!
_STOPPING=0

_cleanup_server() {
    if [[ "$_STOPPING" == "1" ]]; then
        return 0
    fi
    _STOPPING=1
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo "==> Stopping ElasticShelf server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}

if [[ "$_SOURCED" == "0" ]]; then
    trap '_cleanup_server; exit 0' INT TERM
    trap '_cleanup_server' EXIT
else
    trap '_cleanup_server; trap - INT TERM' INT TERM
fi

# Wait up to 60s for server health check before launching browser
READY=0
for _ in $(seq 1 120); do
    if [[ "$_STOPPING" == "1" ]]; then
        break
    fi
    if curl -sf -o /dev/null --max-time 2 "${URL}/api/v1/sim/status" 2>/dev/null; then
        READY=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        [[ "$_STOPPING" == "1" ]] && break
        _exit 1 "Server process exited unexpectedly during startup."
        return 1 2>/dev/null || exit 1
    fi
    sleep 0.5
done

if [[ "$_STOPPING" == "1" ]]; then
    trap - INT TERM
    _exit 0
    return 0 2>/dev/null || exit 0
fi

if [[ "$READY" != "1" ]]; then
    _cleanup_server
    _exit 1 "Server did not respond at $URL within 60s."
    return 1 2>/dev/null || exit 1
fi

echo "==> ElasticShelf is live at $URL"
if [[ "$OPEN_BROWSER" == "1" ]]; then
    echo "==> Launching browser at $URL"
    open_browser "$URL" 2>/dev/null || true
fi
echo "==> Press Start (or Step +1h) in the UI. Press Ctrl-C here to stop the server."

wait "$SERVER_PID" 2>/dev/null || true
if [[ "$_SOURCED" == "1" ]]; then
    trap - INT TERM
    echo "==> Server stopped (virtualenv remains active: $VIRTUAL_ENV)"
fi
