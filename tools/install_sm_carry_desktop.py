#!/usr/bin/env python3
"""Add a local simulation launcher to the NUC desktop (does not start it)."""
import os
from pathlib import Path
import shlex
import subprocess

workspace = Path(os.environ.get('SM_CARRY_WORKSPACE', str(Path.home()/'sm_carry_box_ws'))).resolve()
helper = workspace/'src/motion_tracking_controller/tools/run_sm_carry_sim.sh'
assets = workspace/'assets_sm_carry/full203'
if not helper.is_file() or not (assets/'actor.onnx').is_file():
    raise SystemExit('Build the workspace and prepare full203 assets first')
runner = workspace/'run_sm_carry_desktop.sh'
runner.write_text('#!/usr/bin/env bash\nexec bash '+shlex.quote(str(helper))+' '+shlex.quote(str(assets))+'\n')
runner.chmod(0o755)
desktop = Path.home()/'Desktop/SM Carry Box Sim.desktop'
desktop.parent.mkdir(exist_ok=True)
desktop.write_text('[Desktop Entry]\nType=Application\nName=SM Carry Box 仿真\n'
                   'Comment=G1 SuperMimic carry reference in MuJoCo (no physical box)\n'
                   f'Exec=gnome-terminal -- "{runner}"\n'
                   'Icon=applications-engineering\nTerminal=false\nCategories=Science;Robotics;\n')
desktop.chmod(0o755)
env=os.environ.copy()
env.setdefault('DBUS_SESSION_BUS_ADDRESS', f'unix:path=/run/user/{os.getuid()}/bus')
subprocess.run(['gio','set',str(desktop),'metadata::trusted','true'],env=env,check=False)
print(desktop)
