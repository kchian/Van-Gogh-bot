import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm
import time
import cv2
import atexit
import os
import pickle
import argparse
import sys

from pkg_resources import require
sys.path.insert(1, './pyKinectAzure')
import pyKinectAzure.pykinect_azure as pykinect
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, required=True)
    parser.add_argument('--force', '-f', action='store_true', required=True)
    parser.add_argument('--time', '-t', type=float, default=10)
    parser.add_argument('--dt', '-dt', type=float, default=0.01)
    parser.add_argument('--save_image_seconds', '-sis', type=float, default=3)
    parser.add_argument('--save_image_final', '-sif', action='store_true')
    args = parser.parse_args()

    if os.path.exists(args.name):
        if args.force:
            shutil.rmtree(args.name)
        else:
            print(f"{args.name} exists! aborting")
            sys.exit(-1)
    os.mkdir(args.name)

    if args.save_image_final or args.save_image_seconds:
        pykinect.initialize_libraries()

        # Modify camera configuration
        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        # print(device_config)

        # Start device
        device = pykinect.start_device(config=device_config)
        # cv2.namedWindow('Color Image',cv2.WINDOW_NORMAL)

    out = {}
    out['pose'] = []
    out['joints'] = []
    out['joint_velocities'] = []
    out['t'] = []
    out['dt'] = args.dt
    def save():
        if args.save_image_final:
            capture = device.update()
            # Get the color image from the capture
            ret, color_image = capture.get_color_image()
            # Plot the image
            cv2.imwrite(f"{args.name}/final.png", color_image)
        if len(out["pose"]) > 1:
            num = 0
            with open(f"{args.name}/demo{num}.pkl", 'wb+') as f:
                pickle.dump(out, f)

    atexit.register(save)

    fa = FrankaArm()

    print(f"Guide mode started for {args.time} seconds")
    fa.run_guide_mode(args.time, block=False)


    wait = input("Press enter to start")

    for i in range(int(args.time // args.dt)):
        print(f"{len(out['pose'])}")
        if (i * args.dt) % args.save_image_seconds < args.dt and args.save_image_seconds > 0:
            capture = device.update()
            # Get the color image from the capture
            ret, color_image = capture.get_color_image()
            # Plot the image
            cv2.imwrite(f"{args.name}/{int(i * args.dt)}.png", color_image)

        out["pose"].append(fa.get_robot_state()['pose'].translation)
        out["joints"].append(fa.get_robot_state()['joints'])
        out["joint_velocities"].append(fa.get_robot_state()['joint_velocities'])
        out["t"].append(time.time())
        time.sleep(out['dt'])
    save()
        

