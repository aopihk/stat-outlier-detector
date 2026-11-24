#!/bin/zsh
# Install (or remove) the daily launchd job.
#   scripts/install_launchd.sh            install, runs daily at 08:00 local time
#   scripts/install_launchd.sh 7 30       install, runs daily at 07:30
#   scripts/install_launchd.sh uninstall  remove
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.$(whoami).nba-outliers"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
BIN="$REPO/.venv/bin/nba-outliers"

if [[ "${1:-}" == "uninstall" ]]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Removed $LABEL"
  exit 0
fi

HOUR="${1:-8}"
MINUTE="${2:-0}"

if [[ ! -x "$BIN" ]]; then
  echo "Missing $BIN. Run: python3 -m venv .venv && .venv/bin/pip install -e ." >&2
  exit 1
fi

mkdir -p "$REPO/logs" "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$BIN</string>
    <string>run</string>
    <string>--wait-for-data</string>
  </array>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>$HOUR</integer>
    <key>Minute</key><integer>$MINUTE</integer>
  </dict>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin</string>
  </dict>
  <key>StandardOutPath</key><string>$REPO/logs/launchd.out.log</string>
  <key>StandardErrorPath</key><string>$REPO/logs/launchd.err.log</string>
</dict>
</plist>
PLIST

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
printf "Installed %s: daily at %02d:%02d\n" "$LABEL" "$HOUR" "$MINUTE"
echo "Test it now with: launchctl kickstart gui/$(id -u)/$LABEL"
