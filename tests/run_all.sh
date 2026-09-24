#!/usr/bin/env bash
# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See LICENSE.
#
# Run every test. Headless: no display required, nothing writes outside /tmp.
#
#   ./tests/run_all.sh          # normal run, then again with OCHRE_ACCEL=0
#   ./tests/run_all.sh -q       # one line per test
#
# The suite runs TWICE on purpose. The second pass forces the pure-numpy
# fallback, so it is a first-class tested path rather than an untested
# contingency for machines without a compiler.

set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1

QUIET=0
[ "${1:-}" = "-q" ] && QUIET=1

# Prefer the project venv, fall back to system python.
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi

# Qt tests must never need a real display.
export QT_QPA_PLATFORM=offscreen

# Treat numeric warnings as failures. In a codebase built on exact integer
# arithmetic, an overflow or divide-by-zero warning is a bug report: numpy's
# np.where evaluates BOTH branches, so a formula that merely happens to
# select the safe one still computes the unsafe one. Catching that here is
# how it stays caught.
PYFLAGS="-W error::RuntimeWarning"

pass=0
fail=0
skip=0
failed_names=""

run_pass() {
    local label="$1"
    echo
    echo "=== $label ==="
    for t in tests/test_*.py; do
        [ -e "$t" ] || continue
        name=$(basename "$t" .py)
        out=$("$PY" $PYFLAGS "$t" 2>&1)
        rc=$?
        if [ $rc -eq 0 ]; then
            if echo "$out" | grep -q "^ok: skipped"; then
                skip=$((skip + 1))
                printf '%-28s SKIP\n' "$name"
            else
                pass=$((pass + 1))
                printf '%-28s OK\n' "$name"
            fi
            [ $QUIET -eq 0 ] && echo "$out" | sed 's/^/    /'
        else
            fail=$((fail + 1))
            failed_names="$failed_names $label/$name"
            printf '%-28s FAIL\n' "$name"
            echo "$out" | sed 's/^/    /'
        fi
    done
}

run_pass "default backend"

# Second pass: force the numpy fallback. Skipped if nothing reads the flag yet.
OCHRE_ACCEL=0 run_pass "OCHRE_ACCEL=0 (numpy fallback)"

echo
echo "passed=$pass failed=$fail skipped=$skip"
if [ $fail -ne 0 ]; then
    echo "failing:$failed_names"
fi
[ $fail -eq 0 ]
