#!/usr/bin/env python3
"""Capture an isolated native MuJoCo/ROS2 run with Xvfb and ffmpeg."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def frame_window(display, screenshot):
    x=ctypes.CDLL('libX11.so.6');t=ctypes.CDLL('libXtst.so.6');P=ctypes.c_void_p;U=ctypes.c_ulong
    x.XOpenDisplay.argtypes=[ctypes.c_char_p];x.XOpenDisplay.restype=P
    x.XDefaultRootWindow.argtypes=[P];x.XDefaultRootWindow.restype=U
    x.XQueryTree.argtypes=[P,U,ctypes.POINTER(U),ctypes.POINTER(U),ctypes.POINTER(ctypes.POINTER(U)),ctypes.POINTER(ctypes.c_uint)]
    x.XMoveResizeWindow.argtypes=[P,U,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint]
    x.XKeysymToKeycode.argtypes=[P,U];x.XKeysymToKeycode.restype=ctypes.c_uint
    x.XFlush.argtypes=[P];x.XCloseDisplay.argtypes=[P]
    t.XTestFakeKeyEvent.argtypes=[P,ctypes.c_uint,ctypes.c_int,U]
    d=x.XOpenDisplay(display.encode())
    if not d:raise RuntimeError('Xvfb display unavailable')
    try:
        root=x.XDefaultRootWindow(d);a,b=U(),U();children=ctypes.POINTER(U)();n=ctypes.c_uint()
        x.XQueryTree(d,root,ctypes.byref(a),ctypes.byref(b),ctypes.byref(children),ctypes.byref(n))
        for i in range(n.value):x.XMoveResizeWindow(d,children[i],0,0,1280,800)
        x.XFlush(d);time.sleep(.3)
        def key(sym,on):
            t.XTestFakeKeyEvent(d,x.XKeysymToKeycode(d,sym),on,0);x.XFlush(d);time.sleep(.12)
        key(0xff09,1);key(0xff09,0)
        key(0xffe1,1);key(0xff09,1);key(0xff09,0);key(0xffe1,0)
        key(ord(']'),1);key(ord(']'),0)
        time.sleep(.3)
        from PIL import ImageGrab
        ImageGrab.grab(xdisplay=display).save(screenshot)
    finally:x.XCloseDisplay(d)


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--display',default=':93');p.add_argument('--startup-delay',type=float,default=6.)
    a=p.parse_args();assets=a.assets.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise RuntimeError('Choose a new output directory')
    env=os.environ.copy();env['DISPLAY']=a.display;env['ROS_LOG_DIR']=str(out/'ros_logs')
    processes=[];logs=[]
    def launch(cmd,name):
        log=(out/(name+'.log')).open('w');logs.append(log)
        proc=subprocess.Popen(cmd,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(proc);return proc
    xvfb=video=observer=sim=None
    try:
        xvfb=launch(['Xvfb',a.display,'-screen','0','1280x800x24','-nolisten','tcp'],'xvfb');time.sleep(1)
        if xvfb.poll() is not None:raise RuntimeError('Xvfb failed; select an unused display')
        video=launch(['ffmpeg','-y','-loglevel','warning','-f','x11grab','-framerate','25','-video_size','1280x800','-i',a.display,
                      '-c:v','libx264','-preset','ultrafast','-crf','22','-pix_fmt','yuv420p',str(out/'raw_capture.mp4')],'ffmpeg')
        observer=launch(['/usr/bin/python3',str(Path(__file__).with_name('sm_carry_observe.py')),'--output',str(out)],'observer')
        sim=launch(['ros2','launch','motion_tracking_controller','sm_carry_mujoco.launch.py',f'assets:={assets}',
                    f'controller_start_delay:={a.startup_delay}'],'simulation')
        time.sleep(3)
        if sim.poll() is not None:raise RuntimeError('ROS launch exited; inspect simulation.log')
        try:frame_window(a.display,out/'initial_view.png')
        except Exception as e:(out/'framing_error.txt').write_text(str(e))
        observer.wait(timeout=125)
    finally:
        for proc in [sim,observer,video]:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGINT)
                try:proc.wait(timeout=6)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        if xvfb is not None and xvfb.poll() is None:xvfb.terminate();xvfb.wait(timeout=5)
        for log in logs:log.close()
    if (out/'result.json').exists():print((out/'result.json').read_text())
    if (out/'raw_capture.mp4').exists():
        # Keep ~1 second of the prepared initial pose and the full physics run.
        subprocess.run(['ffmpeg','-y','-loglevel','error','-ss',str(max(0,a.startup_delay-.5)),
            '-i',str(out/'raw_capture.mp4'),'-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(out/'sim2sim.mp4')],check=True)


if __name__=='__main__':main()
