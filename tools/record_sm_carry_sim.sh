#!/usr/bin/env bash
set -eo pipefail
helper_dir=$(cd -- "$(dirname -- "$0")" && pwd)
source "$helper_dir/sm_carry_env.sh"
if [ "$#" -ne 2 ]; then echo "Usage: $0 PREPARED_ASSET_DIRECTORY OUTPUT_DIRECTORY" >&2; exit 2; fi
exec python3 "$helper_dir/record_sm_carry_sim.py" --assets "$1" --output "$2"
