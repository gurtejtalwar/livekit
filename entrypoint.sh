#!/bin/sh
set -e

echo "Downloading files..."
python server.py download-files

echo "Starting dev server..."
exec python server.py dev