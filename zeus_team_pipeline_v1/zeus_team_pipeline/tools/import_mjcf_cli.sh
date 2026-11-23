#!/usr/bin/env bash
# Run inside a 6.1 Isaac Sim container that has standalone_examples.
# Usage: ./import_mjcf_cli.sh /absolute/path/robot.xml /absolute/output/directory
set -euo pipefail
if [ "$#" -ne 2 ]; then echo 'Usage: import_mjcf_cli.sh ABSOLUTE_MJCF ABSOLUTE_OUTPUT_DIR'; exit 2; fi
SOURCE="$1"; TARGET="$2"
[[ -f "$SOURCE" ]] || { echo "Missing model: $SOURCE"; exit 2; }
[[ "$SOURCE" = /* && "$TARGET" = /* ]] || { echo 'Use absolute paths'; exit 2; }
mkdir -p "$TARGET"
# Prefer user's verified Isaac Sim install. This script does not start separate Kit while GUI is active.
CANDIDATES=("/opt/isaac-sim" "/isaac-sim" "/root/isaac-sim")
FOUND=''
for path in "${CANDIDATES[@]}"; do
 if [[ -f "$path/python.sh" && -f "$path/standalone_examples/api/isaacsim.asset.importer.mjcf/mjcf_import.py" ]]; then FOUND="$path";break;fi
done
if [[ -z "$FOUND" ]];then
 echo 'Standalone importer not found in common paths. Use Isaac Sim GUI File > Import instead.'
 echo 'Do not execute this script with system Python.'
 exit 3
fi
cd "$FOUND"
./python.sh standalone_examples/api/isaacsim.asset.importer.mjcf/mjcf_import.py --mjcf "$SOURCE" --usd-path "$TARGET" --debug-mode
