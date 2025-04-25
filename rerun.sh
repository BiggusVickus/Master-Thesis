#!/bin/bash

FILE="streamlit.py"
RUN_CMD="./.venv/bin/python /Users/vicpi/Documents/GitHub/Master_Thesis/streamlit.py"

echo "📡 Watching $FILE. Will rerun on crash and change..."

while true; do
    $RUN_CMD
    echo "⚠️  $FILE crashed or exited. Waiting for changes before rerun..."
    sleep 1
    # fswatch blocks until file is changed
    # fswatch -1 "$FILE"
done
