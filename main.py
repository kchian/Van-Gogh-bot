from multiprocessing.sharedctypes import Value
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import learn
import vis
import math
import img_lib
import argparse
import sys
sys.path.insert(1, './pyKinectAzure')

import pyKinectAzure.pykinect_azure as pykinect
import cv2
import time


import subprocess

TOP_LEFT = [ 0.32886487, -0.25668124]
MID_LEFT = [ 0.32886487, -0.25668124] # UP-DOWN, LEFT-RIGHT
BOTTOM_RIGHT = [ 0.59114306,  0.28672476]

# Shouldn't be changing these but...
W = BOTTOM_RIGHT[1] - TOP_LEFT[1]
H = BOTTOM_RIGHT[0] - TOP_LEFT[0]
MID_LEFT[1] += W / 2 # cut canvas in half to draw on one side

import numpy as np

from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

from frankapy.utils import min_jerk, min_jerk_weight

import rospy

def flip_coords(traj, point):
    flipped_coords = np.array([[point[i] + point[i] - coord[i] for i in range(len(coord))] for coord in traj])
    return flipped_coords


def run_trajectory(fa, pose_traj):
    rospy.loginfo('Generating Trajectory')

    init = pickle.load(open('test.pkl','rb'))

    T = 15
    dt = 0.01
    ts = np.arange(0, T, dt)

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=10)
    rate = rospy.Rate(1 / dt)

    rospy.loginfo('Publishing pose trajectory...')
    # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
    fa.goto_pose(init, duration=T, dynamic=True, buffer_time=10,
        cartesian_impedances=[600.0, 600.0, 600.0, 50.0, 50.0, 50.0]
    )
    quaternion = init.quaternion
    
    init_time = rospy.Time.now().to_time()
    for i in range(2, len(ts)):
        timestamp = rospy.Time.now().to_time() - init_time
        traj_gen_proto_msg = PosePositionSensorMessage(
            id=i, timestamp=timestamp,
            position=pose_traj[i], 
            quaternion=quaternion
		)
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
            )

        # rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
        pub.publish(ros_msg)
        rate.sleep()

    # Stop the skill
    # Alternatively can call fa.stop_skill()
    term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time, should_terminate=True)
    ros_msg = make_sensor_group_msg(
        termination_handler_sensor_msg=sensor_proto2ros_msg(
            term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
        )
    pub.publish(ros_msg)

    rospy.loginfo('Done')

def simple_goto(pos, duration):
    for i in range(3):
        try:
            des_pose = RigidTransform(
                rotation=np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1],
                ]),
                translation=np.array(pos),
                from_frame='franka_tool',
                to_frame='world',
            )
            fa.goto_pose(des_pose, duration=duration, use_impedance=False)
            break
        except ValueError as e:
            if e.message == 'Cannot send another command when the previous skill is active!':
                print(f"Still executing. Sleeping 10, try {i}")
                time.sleep(10)
                continue
            else:
                raise
            

    


if __name__ == "__main__":
    # Load in strokes library
    parser = argparse.ArgumentParser()
    parser.add_argument('--dmp_library', '-d', type=str, default="strokes/dmp.pkl")
    parser.add_argument('--image_path', '-i', type=str, default="")
    parser.add_argument('--time', '-T', type=float, default=60, help="Time per stroke")
    parser.add_argument('--execute', '-e', action='store_true')
    parser.add_argument('--draw_height', '-dh', type=float, default=0.08)
    args = parser.parse_args()
    
    if args.execute:
        fa = FrankaArm()
        fa.reset_joints()
        fa.close_gripper()

    # Load stroke library
    with open(args.dmp_library, "rb") as f:
        stroke_library = pickle.load(f)
        print(f"Stroke library loaded from {args.dmp_library}")

    # if there is a picture already, use that
    image_path = args.image_path
    if args.image_path == "":
        pykinect.initialize_libraries()
        # Modify camera configuration
        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        # Start device
        device = pykinect.start_device(config=device_config)

        # Take picture of board, crop to half probably, set anchor
        time.sleep(1)

        capture = device.update()
        # Get the color image from the capture
        ret, color_image = capture.get_color_image()
        # Plot the image
        image_path = "last_board.png"
        cv2.imwrite(image_path, color_image)
        print("saved image as " + image_path)


    # Subprocess Popen Faster RCNN of picture, write to temp file
    command = ["python3.9", "rcnn_main.py", image_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    with open(f"strokes/dmp.pkl", "rb") as f:
        dmp_weights = pickle.load(f)

    # Take in strokes / bounding boxes from file
    with open("model_out.txt") as f:
        print(f"Read model outputs:")
        for line in f:
            print(line)
            s = line.split(" ")
            stroke = s[0]
            bbox = [int(i) for i in s[1:]]
            # I have no idea what the size or starting point of my strokes are lol
            # I guess I could keep it in the dmp dict? try to map that to a specific point?
    
            # translate bounding box relative to my anchor (v1: just use center)
            center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
            # So (0, 0) = TOP_LEFT, img_lib.H = 290 rn
            x = MID_LEFT[0] + center[1] / (img_lib.W/2) * (W/2)
            y = MID_LEFT[1] + center[0] / img_lib.H * H
            print(f"Drawing at {x}, {y}")

            # get trajectory
            initial_state = [x, y, 0.2]
            down = [x, y, args.draw_height]
            if x < MID_LEFT[0] or y < MID_LEFT[1] or x > BOTTOM_RIGHT[0] or y > BOTTOM_RIGHT[1]:
                print(f"Detected object is out of range. Skipping.")
                continue
            if stroke == "plus":
                continue # Its too messed up lol


            t = np.linspace(0, args.time, 1000)
            trajectory = []
            for axis in range(3):
                axis_goal = down[axis] + dmp_weights[stroke]["orig_x0"][axis] - dmp_weights[stroke]["orig_g"][axis]
                weights = dmp_weights[stroke]["weights"]
                axis_trajectory = learn.get_pos_from_f(
                    t, 
                    down[axis], 
                    axis_goal, 
                    weights["mu"], 
                    weights["h"], 
                    weights["weights"][axis][0], 
                    alpha=weights["alpha"], 
                    beta=weights["beta"],
                    solver="odeint",
                )
                trajectory.append(axis_trajectory)
            trajectory = np.stack(trajectory).astype(np.float32).T

            trajectory = flip_coords(trajectory, down)
            # vis.plot_drawing(trajectory)
            # plt.show()

            # Modify trajectory for failsafe t okeep it within canvas bounds
            
            mask = np.where(trajectory[:,0] < MID_LEFT[0], True, False)
            trajectory[mask, 0] = MID_LEFT[0]
            mask = np.where(trajectory[:,1] < MID_LEFT[1], True, False)
            trajectory[mask, 1] = MID_LEFT[1]
            mask = np.where(trajectory[:,0] > BOTTOM_RIGHT[0], True, False)
            trajectory[mask, 0] = BOTTOM_RIGHT[0]
            mask = np.where(trajectory[:,1] > BOTTOM_RIGHT[1], True, False)
            trajectory[mask, 1] = BOTTOM_RIGHT[1]

            # axis_trajectory[:,0][axis_trajectory[:,0] < TOP_LEFT[0]] = TOP_LEFT[0]
            # axis_trajectory[:,1][axis_trajectory[:,1] < TOP_LEFT[1]] = TOP_LEFT[1]
            # axis_trajectory[:,0][axis_trajectory[:,0] > BOTTOM_RIGHT[0]] = BOTTOM_RIGHT[0]
            # axis_trajectory[:,1][axis_trajectory[:,1] > BOTTOM_RIGHT[1]] = BOTTOM_RIGHT[1]


            if args.execute:
                fa.close_gripper()
                simple_goto(initial_state, 10)
                simple_goto(down, 3)
                try:
                    run_trajectory(fa, trajectory)
                except IndexError:
                    print("Done with one trajectory!")
                fa.stop_skill()
                fa.close_gripper()

                # lift pen
                cur_loc = fa.get_robot_state()['pose'].translation
                cur_loc[-1] = 0.2
                simple_goto(cur_loc, 3)
            else:

            # Move to close to board (y = 0.2) over the initial place
                vis.plot_path(trajectory)
                plt.show()

        if args.execute:
            print("Done with all trajectories!")
            fa.reset_joints()
