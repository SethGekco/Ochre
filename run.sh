#!/usr/bin/env bash
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
#
# Launch the editor. Safe to call from anywhere, which is the point: a
# desktop entry runs with whatever working directory the session feels like
# handing it, so this resolves its own location rather than assuming one.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

python="$here/.venv/bin/python"
if [[ ! -x "$python" ]]; then
    # No virtualenv: fall back rather than refusing to start. The engine
    # needs only numpy and Pillow, and a system Python usually has both.
    python="$(command -v python3 || true)"
    if [[ -z "$python" ]]; then
        echo "ochre: no python3 found" >&2
        exit 1
    fi
fi

# From a terminal, behave like a normal command. From a desktop icon there
# is nowhere for a traceback to go, so keep one -- appended, because the
# interesting run is usually not the last one.
if [[ -t 1 ]]; then
    exec "$python" -m ochre.ui "$@"
fi

log="${XDG_STATE_HOME:-$HOME/.local/state}/ochre"
mkdir -p "$log"
exec "$python" -m ochre.ui "$@" >>"$log/ochre.log" 2>&1
