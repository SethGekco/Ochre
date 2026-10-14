#!/usr/bin/env bash
# Install a desktop entry for Ochre into the current user's session.
#
# Paths are written ABSOLUTE at install time rather than relying on the
# binary being on PATH, because Ochre is normally run from a checkout.
# Re-run this after moving the checkout.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$apps"

entry="$apps/ochre.desktop"
sed "s|@ROOT@|$root|g" "$root/packaging/ochre.desktop" > "$entry"
chmod +x "$entry"
echo "installed $entry"

# A copy on the desktop, if there is one. KDE and GNOME both require the
# file to be executable before they will treat it as a launcher rather than
# as a text file to open.
desktop="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
if [[ -d "$desktop" ]]; then
    cp "$entry" "$desktop/ochre.desktop"
    chmod +x "$desktop/ochre.desktop"
    # KDE will not launch a desktop file it has not been told to trust.
    if command -v kwriteconfig6 >/dev/null 2>&1 || command -v gio >/dev/null 2>&1; then
        gio set "$desktop/ochre.desktop" metadata::trusted true 2>/dev/null || true
    fi
    echo "installed $desktop/ochre.desktop"
fi

command -v update-desktop-database >/dev/null 2>&1 \
    && update-desktop-database "$apps" 2>/dev/null || true
echo "done"
