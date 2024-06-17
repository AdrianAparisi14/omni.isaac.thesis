#launch Isaac Sim before any other imports
#default first two lines in any standalone application
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True}) # we can also run as headless.

import copy
import sys
import time
import numpy as np
import quaternion
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
from pytransform3d.transform_manager import TransformManager
from omni.sdu.utils.utilities import plot as plt
from omni.isaac.core.utils import rotations as r
from scipy.spatial.transform import Rotation as R
import csv
from datetime import datetime
import json
import argparse

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_io import RTDEIOInterface as RTDEIo

from omni.sdu.utils.utilities.math_util import wrench_trans, get_robot_pose_as_transform, get_pose_as_transform, get_transform_as_pq, \
                                                                get_transform_as_ur_pose, R_to_rotvec, get_transform_as_pose, get_transform_as_ur_pose_rotvec
from omni.sdu.utils.utilities.admittance_controller_position import AdmittanceControllerPosition
from omni.sdu.ai.utilities import utils as ut_ai

from omni.isaac.thesis.force_dataset.utils.lowpass_filter import LowPassFilter

PLACE_POSE = [0.28632085376286964, 0.363607215293683, 0.2263943406539749, 2.8975001898637833, -1.2122370871517927, 0.0022033002335769235]
PLACE_Q = [1.1950205564498901, -1.6847688160338343, -2.1948113441467285, -0.8513174814036866, 1.5688886642456055, 0.41890692710876465]

PLACE_APPROX_POSE = [0.2863001578683557, 0.36359806537224587, 0.325, 2.897516032346434, -1.2120822261580186, 0.002188515010319512]
PLACE_APPROX_Q = [1.1951044797897339, -1.5381060403636475, -2.048758029937744, -1.1440003675273438, 1.56956148147583, 0.4188830852508545]

PICK_POSE = [0.5105244047089075, 0.13011966784806409, 0.2318826285091872, 2.897222744351678, -1.2125884065595494, 0.0024652353585882325]
PICK_Q = [0.5050456523895264, -1.8133603535094203, -2.0161633491516113, -0.9003079694560547, 1.568648338317871, -0.2712205092059534]

PICK_APPROX_POSE = [0.5105081441259004, 0.1301499977456452, 0.38617096816437335, 2.897241503746813, -1.2124379562427063, 0.002408788550950928]
PICK_APPROX_Q = [0.5052253007888794, -1.6420532665648402, -1.777040958404541, -1.3107908529094239, 1.5694770812988281, -0.27130395570863897]

class RobotRTDE(RTDEControl, RTDEReceive, RTDEIo):
    def __init__(self, ip: str):
        self.ip = ip
        rtde_frequency = 500.0
        RTDEControl.__init__(self, ip, rtde_frequency, flags=RTDEControl.FLAG_VERBOSE | RTDEControl.FLAG_UPLOAD_SCRIPT)
        RTDEReceive.__init__(self, ip, -1., [])
        RTDEIo.__init__(self, ip)

def get_hole_target(pose, orientation, timestep):
    # Pose
    hole_target_pose = copy.deepcopy(pose)
    hole_target_pose[2] = hole_target_pose[2] - timestep/80
    # print("hole_target_pose[2]: ", hole_target_pose[2])
    if hole_target_pose[2] < 0.0300:   #0.0335:
        hole_target_pose[2] = 0.0300 
    
    # Orientation
    # Original quaternion
    original_quaternion = copy.deepcopy(orientation)

    # Axis of rotation (normalized) and the angle in radians
    axis_of_rotation = np.array([0, 0, 1])
    angle_in_radians = timestep/4

    # Create a quaternion representing the rotation
    rotation_quaternion = pr.quaternion_from_axis_angle(np.concatenate((axis_of_rotation, [angle_in_radians])))

    # Rotate the original quaternion
    rotated_quaternion = pr.concatenate_quaternions(rotation_quaternion, original_quaternion)
    
    return hole_target_pose, orientation


def main():
    """Main entry point allowing external calls

    Args:
      args ([str]): input can be either "train" or "validation". "train" will save the files in a folder that will be used in other programs to train the
                    classification method; "validation" must be used for validate novel data.
    """
    
    # try:
    #     mode = sys.argv[1]
    #     print("RUNNING IN MODE: ", mode)
    # except ValueError as e:
    #     print(f"Error: {e}")
    #     print("Maybe forgot to write either 'train' or 'validation'?")
    #     sys.exit(1)  # Terminate the program 
        
    args = parse_arguments()
    model = args.model
    mode = args.mode
        
    frequency = 500.0  # Hz
    dt = 1 / frequency
    vel = 0.4
    acc = 0.5
    lookahead_time = 0.1
    gain = 1000

    # Initialize transform manager
    tm = TransformManager()
    
    # Initialize low pass filter:
    lp_filter = LowPassFilter()
    lp_filter.reset()

    print("Initializing robot...")
    robot_custom_script_file_path = "/home/asegui/CustomScript/rtde_custom_control.script"
    try:
        ur_robot = RobotRTDE("192.168.1.14")
        ur_robot.setCustomScriptFile(robot_custom_script_file_path)
    except Exception as e:
        print("Failed to connect with the robot...\n " + str(e))
    
    # Zero the force sensor
    print("Zeroing FT sensor...")
    ur_robot.zeroFtSensor()
    # wait for the first data to be received, before initializing and starting the controller
    time.sleep(0.5)

    # Move robot to the starting position
    print("Moving robot to the start position...")
    # ur_robot.moveJ(Q_UP_BASE) # q up of the base
    ur_robot.moveJ(PICK_APPROX_Q) # Record pose
    time.sleep(2)
    
    print("\n\nPerforming pick and place operation ...")
    velocity = 0.1
    acceleration = 0.1
    
    # Move to pick approx. pose
    ur_robot.moveL(PICK_APPROX_POSE, velocity, acceleration) # Record pose
    
    # Activate the gripper
    recipe = 7
    ur_robot.setInputIntRegister(18, recipe)
    time.sleep(0.1)
    ur_robot.setInputIntRegister(19, 2) # Release position of the gripper
    time.sleep(0.5)
    
    # Pick pose
    ur_robot.moveL(PICK_POSE, velocity, acceleration) # Record pose
    
    # Close the gripper
    ur_robot.setInputIntRegister(18, recipe)
    time.sleep(0.1)
    ur_robot.setInputIntRegister(19, 1) # Grasp position of the gripper
    time.sleep(1.5)
    

    # Send a linear path with blending in between
    ur_robot.moveL(PICK_APPROX_POSE, velocity, acceleration) # Record pose

    ur_robot.moveJ(PLACE_APPROX_Q)
    
    time.sleep(5)
    
    # Retrieve Pose of the robot in the starting position
    robot_pose = ur_robot.getActualTCPPose() # [-0.4338166342180918, -0.21722334966328855, 0.2768509807303988, 3.0574024796186383, -0.7080677813361, -0.0026293921812557504] # harcoded for debugging purposes :D
    # print("robot_pose_start: ", robot_pose)
    orient_rotvec = R.from_rotvec(robot_pose[3:])
    robot_tcp_pose = (robot_pose[:3],orient_rotvec.as_matrix())
    # print("robot_tcp_pose: ", robot_tcp_pose)
    
    T_base_tcp = get_pose_as_transform(robot_tcp_pose)
    # print("T_base_tcp: ", T_base_tcp)

    # define a tip wrt. the TCP frame
    quat_identity = quaternion.as_float_array(quaternion.quaternion(1.0, 0.0, 0.0, 0.0))
    T_tcp_tip = pt.transform_from_pq(np.hstack((np.array([0, 0, 0.2]), quat_identity)))
    T_tip_tcp = np.linalg.inv(T_tcp_tip)
    T_base_tip = T_base_tcp @ T_tcp_tip

    # get tip in base as position and quaternion
    T_base_tip_pq = pt.pq_from_transform(T_base_tip)
    T_base_tip_pos_init = T_base_tip_pq[0:3]
    T_base_tip_quat_init = T_base_tip_pq[3:7]

    # Set initial circle target
    counter = 0.0
    x_desired, orient_desired = get_hole_target(T_base_tip_pos_init, T_base_tip_quat_init, counter)
    T_base_tip_circle = pt.transform_from_pq(np.hstack((x_desired, T_base_tip_quat_init)))
    T_base_tcp_circle = T_base_tip_circle @ T_tip_tcp
    # print("T_base_tip_circle: ", T_base_tcp_circle)

    # Use moveL to move to the initial point on the circle.
    
    #print("ur_robot.moveL(get_transform_as_ur_pose(T_base_tcp_circle)): ", get_transform_as_ur_pose_rotvec(T_base_tcp_circle))

    print("Initializing AdmittanceController...")
    adm_controller = AdmittanceControllerPosition(start_position=x_desired, start_orientation=T_base_tip_quat_init,
                                                  start_ft=ur_robot.getActualTCPForce())
    print("Starting AdmittanceController!")
    
    # The controller parameters can be changed, the following parameters corresponds to the default,
    # and we set them simply to demonstrate that they can be changed.
    adm_controller.M = np.diag([22.5,22.5,22.5])# 22.5, 22.5, 22.5])
    adm_controller.D = np.diag([4000, 4000, 4000]) # 160, 160, 160])
    adm_controller.K = np.diag([30, 30, 30])# [54, 54, 54])

    adm_controller.Mo = np.diag([0.25, 0.25, 0.25])
    adm_controller.Do = np.diag([200, 200, 200])
    adm_controller.Ko = np.diag([7, 7, 7]) # [10, 10, 10])
    
    # Setup plot data
    plot = plt.plot_juggler()
    
    # Setup write data on a csv
    data_to_store = []
    batch_size = 0
    current_time = datetime.now()
    csv_file_path = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_pick_place/real/admittance/data" + str(current_time) + ".csv"
    csv_validation = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/script/csv_validation_pick_place/data" + str(current_time) + ".csv"
    
    # Read init pose to start the loop
    robot_pose = ur_robot.getActualTCPPose()
    # Zero the force sensor
    print("Zeroing FT sensor...")
    ur_robot.zeroFtSensor()
    # wait for the first data to be received, before initializing and starting the controller
    time.sleep(0.5)
    
    try:
        start_time_T = time.time()
        while counter<=9: #and robot_pose[2]>POSE_IN_BASE[2]:
            start_time_ = time.time()
            start_time = ur_robot.initPeriod()
            
            # get current robot pose
            robot_pose = ur_robot.getActualTCPPose()
            orient_rotvec = R.from_rotvec(robot_pose[3:])
            robot_tcp_pose = (robot_pose[:3],orient_rotvec.as_matrix())
            
            T_base_tcp = get_pose_as_transform(robot_tcp_pose)
            T_base_tip = T_base_tcp @ T_tcp_tip

            # get tip in base as position and quaternion
            T_base_tip_pq = pt.pq_from_transform(T_base_tip)
            T_base_tip_pos = T_base_tip_pq[0:3]
            T_base_tip_quat = T_base_tip_pq[3:7]

            # get current robot force-torque in base (measured at tcp)
            reading_ = ur_robot.getActualTCPForce()
            lp_filter.process(reading_)
            reading = [lp_filter.f_filtered[0],lp_filter.f_filtered[1],lp_filter.f_filtered[2],lp_filter.mu_filtered[0], lp_filter.mu_filtered[1],lp_filter.mu_filtered[2]]
            # reading = [reading_[0],reading_[1],reading_[2],reading_[3],reading_[4],reading_[5]]
            # plot.publish(reading)#.tolist())
            # print("forde read: ", reading)
            f_base = reading[0:3]
            mu_base = reading[3:6]
            
            # If there is no force compensation for the gripper:
            # f_base, mu_base = force_compensation(Fg, Foff, Moff, r, reading, orient_rotvec.as_matrix())

            # rotate forces from base frame to TCP frame (necessary on a UR robot)
            R_tcp_base = np.linalg.inv(T_base_tcp[:3, :3])
            f_tcp = R_tcp_base @ f_base
            mu_tcp = R_tcp_base @ mu_base
            graph = [f_tcp[0],f_tcp[1],f_tcp[2],mu_tcp[0],mu_tcp[1],mu_tcp[2]]
            plot.publish(graph)
            # graph = f_tcp + mu_tcp
            # plot.publish(graph.tolist())

            # use wrench transform to place the force torque in the tip.
            mu_tip, f_tip = wrench_trans(mu_tcp, f_tcp, T_tcp_tip)

            # rotate forces back to base frame
            R_base_tip = T_base_tip[:3, :3]
            f_base_tip = R_base_tip @ f_tip

            # the input position and orientation is given as tip in base
            adm_controller.pos_input = T_base_tip_pos
            adm_controller.rot_input = quaternion.from_float_array(T_base_tip_quat)
            adm_controller.q_input = ur_robot.getActualQ()
            adm_controller.ft_input = np.hstack((f_base_tip, mu_tip))

            # get circle target
            x_desired, orient_desired = get_hole_target(T_base_tip_pos_init, T_base_tip_quat_init, counter)
            adm_controller.set_desired_frame(x_desired, quaternion.from_float_array(orient_desired))

            # step the execution of the admittance controller
            adm_controller.step()
            output = adm_controller.get_output()
            output_position = output[0:3]
            output_quat = output[3:7]

            # rotate output from tip to TCP before sending it to the robot
            T_base_tip_out = pt.transform_from_pq(np.hstack((output_position, output_quat)))
            T_base_tcp_out = T_base_tip_out @ T_tip_tcp
            base_tcp_out_ur_pose = get_transform_as_ur_pose_rotvec(T_base_tcp_out)
            # print("base_tcp_out_ur_pose: ", base_tcp_out_ur_pose)

            # set position target of the robot
            ur_robot.servoL(base_tcp_out_ur_pose, vel, acc, dt, lookahead_time, gain)    
            ur_robot.waitPeriod(start_time)
            
            # write data in a file
            data_to_store.append(graph)
            data_to_store.append(robot_pose)
            
            counter = counter + dt
            
            end_time = time.time()
            elapsed_time = end_time - start_time_
            # print("Loop executed in", elapsed_time, "seconds")
        
        end_time_T = time.time()
        period = end_time_T - start_time_T
        print("period is: ", period)
        ur_robot.servoStop()
        
        ur_robot.setInputIntRegister(18, recipe)
        time.sleep(0.1)
        ur_robot.setInputIntRegister(19, 2) # Release position of the gripper
        time.sleep(0.5)
        
        ur_robot.moveL(PLACE_APPROX_POSE) # Record pose
        
        print("\nTask finished ... !")
        
        if mode == "train":
            batch_write(csv_file_path, data_to_store)
        elif mode == "validation":
            batch_write(csv_validation, data_to_store)
        else:
            print("\n\nThe input mode doesn't correspond to any of the possibilities. Data file not being saved!\n\n")
        
        if mode == "train":
            result = input("Success?: ")
            with open(csv_file_path, 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(result)

    except KeyboardInterrupt:
        adm_controller.stop()
        ur_robot.stop_control()
    
    # if mode == "validation":
    #     ut_ai.verification_assembly(csv_validation, model)


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process options.")
    parser.add_argument('--model', type=str, default="sim", help='Whether to use the model trained on simulation data or real data (sim/real)')
    parser.add_argument('--mode', type=str, default="train", help='Whether to run the program to record data or validate data (train/validation)')
    return parser.parse_args()
            
def batch_write(output_file, input_list):
    with open(output_file, 'w') as file:
        for i in range(0, len(input_list), 2):
            pair = [input_list[i:i+1], input_list[i+1:i+2]]
            file.write(f'{pair[0]}\t{pair[1]}\n')


if __name__ == "__main__":
    main()
