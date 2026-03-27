#!/bin/bash
# Wrapper script for the LaunchAgent — loads secrets from .env then runs the bot.
# Keep this file readable only by you: chmod 700 run_bot.sh

DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/.env"
exec "$DIR/.venv/bin/python3" "$DIR/copybot.py" --config "$DIR/config.yaml"
