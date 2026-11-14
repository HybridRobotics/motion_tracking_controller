#!/usr/bin/env bash
set -eo pipefail
helper_dir=$(cd -- "$(dirname -- "$0")" && pwd)
source "$helper_dir/sm_carry_env.sh"
if [ "$#" -lt 1 ]; then echo "Usage: $0 PREPARED_ASSET_DIRECTORY [launch arguments]" >&2; exit 2; fi
asset_dir=$(realpath "$1"); shift
exec ros2 launch motion_tracking_controller sm_carry_mujoco.launch.py assets:="$asset_dir" "$@"
