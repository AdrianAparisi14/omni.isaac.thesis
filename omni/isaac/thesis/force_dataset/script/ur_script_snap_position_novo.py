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
import argparse


from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

from omni.sdu.utils.utilities.math_util import wrench_trans, get_robot_pose_as_transform, get_pose_as_transform, get_transform_as_pq, \
                                                                get_transform_as_ur_pose, R_to_rotvec, get_transform_as_pose, get_transform_as_ur_pose_rotvec
from omni.sdu.utils.utilities.admittance_controller_position import AdmittanceControllerPosition
from omni.sdu.ai.utilities import utils as ut_ai
from omni.isaac.thesis.force_dataset.plot.plot_data import plot_last_assembly

POSE_UP_BASE = [0.07282149901783823, -0.5708324942427645, 0.2501080993928734, -1.2090314136855016, 2.899594734296001, -8.132030943904366e-06]
Q_UP_BASE = [-1.2104786078082483, -1.888085504571432, -1.8563084602355957, -0.9857393068126221, 1.5706787109375, -0.42830211320985967]
POSE_IN_BASE = [0.07280823373568777, -0.5708499894880407, 0.20250156760131718, 1.2089868581697196, -2.8996130179915816, 0.00013304251266285088]
Q_IN_BASE = [-1.2104786078082483, -1.958468576470846, -1.9049134254455566, -0.8666699689677735, 1.570390224456787, -0.42831403413881475]

# Novo Robot poses
POSE_UP_HOUSING = [-0.6368332753024288, 0.11056229274682902, 0.39146922077074836, -1.2093912154655926, 2.8928644531185927, -0.0036313707994368354]
Q_UP_HOUSING = [-3.1065080801593226, -1.922833105126852, -1.3964296579360962, -1.4017898005298157, 1.5666873455047607, -2.3265751043902796]
POSE_IN_HOUSING = [-0.6352958975019792, 0.11011307106791235, 0.32552746274826766+0.035, -1.2089855387777215, 2.8996190770039196, -4.941977858874225e-05]
Q_IN_HOUSING = [-1.109781567250387, -1.5930754146971644, -1.7648591995239258, -1.3539645981839676, 1.5733470916748047, -0.32892352739443]

class RobotRTDE(RTDEControl, RTDEReceive):
    def __init__(self, ip: str):
        self.ip = ip
        rtde_frequency = 500.0
        RTDEControl.__init__(self, ip, rtde_frequency)
        RTDEReceive.__init__(self, ip, -1., [])

def get_hole_target(pose, orientation, timestep):
    # Pose
    hole_target_pose = copy.deepcopy(pose)
    hole_target_pose[2] = hole_target_pose[2] - timestep/80
    print("hole_target_pose[2]: ", hole_target_pose[2])
    if hole_target_pose[2] < 0.03899102+0.064445019: #0.04350: # 0.03891102: #[Stop early(fail)=0.04385] // [correct assembly=0.03899102] // [Stop late(fail)=0.036021]
        hole_target_pose[2] = 0.03899102+0.064445019 #0.04350 # 0.03891102
    
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

    print("Initializing robot...")
    try:
        ur_robot = RobotRTDE("192.168.1.12")#"192.168.1.14")
    except Exception as e:
        print("Failed to connect with the robot...\n " + str(e))
        
    print("Zeroing FT sensor...")
    ur_robot.zeroFtSensor()
    # wait for the first data to be received, before initializing and starting the controller
    time.sleep(0.5)

    # Move robot to the starting position
    print("Moving robot to the start position...")
    # ur_robot.moveJ(Q_UP_BASE) # q up of the base
    # ur_robot.moveL([0.07282149901783823, -0.5708324942427645, 0.285, -1.2090314136855016, 2.899594734296001, -8.132030943904366e-06]) # pose up of the base
    ur_robot.moveJ(Q_UP_HOUSING)
    time.sleep(2)
        
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
    # print("Moving the robot to the start position of the insertion (should be the same...)")
    # #print("ur_robot.moveL(get_transform_as_ur_pose(T_base_tcp_circle)): ", get_transform_as_ur_pose_rotvec(T_base_tcp_circle))

    # print("Initializing AdmittanceController...")
    # adm_controller = AdmittanceControllerPosition(start_position=x_desired, start_orientation=T_base_tip_quat_init,
    #                                               start_ft=ur_robot.getActualTCPForce())
    # print("Starting AdmittanceController!")
    
    # # time.sleep(20)

    # # The controller parameters can be changed, the following parameters corresponds to the default,
    # # and we set them simply to demonstrate that they can be changed.
    # adm_controller.M = np.diag([22.5,22.5,22.5])# 22.5, 22.5, 22.5])
    # adm_controller.D = np.diag([1000, 1000, 1000]) # 160, 160, 160])
    # adm_controller.K = np.diag([20, 20, 20])# [54, 54, 54])

    # adm_controller.Mo = np.diag([0.25, 0.25, 0.25])
    # adm_controller.Do = np.diag([200, 200, 200])
    # adm_controller.Ko = np.diag([7, 7, 7]) # [10, 10, 10])
    
    # Setup plot data
    # plot = plt.plot_juggler()
    
    # Setup write data on a csv
    data_to_store = []
    batch_size = 0
    current_time = datetime.now()
    
    # csv_file_path = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_real_robot_admittance/data" + str(current_time) + ".csv"
    # csv_validation = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/script/csv_validation/data" + str(current_time) + ".csv"
    
    csv_file_path = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_real_robot_novo/csv_real_robot_position_novo/data" + str(current_time) + ".csv"
    csv_validation = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/script/csv_validation/data" + str(current_time) + ".csv"
    
    
    # Read init pose to start the loop
    robot_pose = ur_robot.getActualTCPPose()
    
    iteration_counter = 0.0
    
    try:
        start_time_T = time.time()
        while iteration_counter <= 3800: #and robot_pose[2]>POSE_IN_BASE[2]:
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
            reading = [reading_[0],reading_[1],reading_[2],reading_[3],reading_[4],reading_[5]]
            # plot.publish(reading)#.tolist())
            # print("forde read: ", reading)
            f_base = reading[0:3]
            mu_base = reading[3:6]

            # rotate forces from base frame to TCP frame (necessary on a UR robot)
            R_tcp_base = np.linalg.inv(T_base_tcp[:3, :3])
            f_tcp = R_tcp_base @ f_base
            mu_tcp = R_tcp_base @ mu_base
            graph = [f_tcp[0],f_tcp[1],-f_tcp[2],mu_tcp[0],mu_tcp[1],mu_tcp[2]]
            # plot.publish(graph)

            # use wrench transform to place the force torque in the tip.
            mu_tip, f_tip = wrench_trans(mu_tcp, f_tcp, T_tcp_tip)

            # rotate forces back to base frame
            R_base_tip = T_base_tip[:3, :3]
            f_base_tip = R_base_tip @ f_tip

            # the input position and orientation is given as tip in base
            # adm_controller.pos_input = T_base_tip_pos
            # adm_controller.rot_input = quaternion.from_float_array(T_base_tip_quat)
            # adm_controller.q_input = ur_robot.getActualQ()
            # adm_controller.ft_input = np.hstack((f_base_tip, mu_tip))

            # get circle target
            x_desired, orient_desired = get_hole_target(T_base_tip_pos_init, T_base_tip_quat_init, counter)
            # [NOVO] stop pose command --
            # robot_pose_ = ur_robot.getActualTCPPose()
            # if robot_pose_[2] <= 0.239507993061658:
            #     x_desired = 0.239507993061658
                
            # print(robot_pose_[2], x_desired)
            # ---
            # orient_desired_ = R.from_quat(orient_desired)
            # print("Desired pos orient: ", x_desired, orient_desired_.as_euler('xyz'))
            # T_base_tip_desired = pt.transform_from_pq(np.hstack((x_desired, orient_desired)))
            # print("T_base_tip_desired: ", T_base_tip_desired)
            # adm_controller.set_desired_frame(x_desired, quaternion.from_float_array(orient_desired))

            # # step the execution of the admittance controller
            # adm_controller.step()
            # output = adm_controller.get_output()
            # output_position = output[0:3]
            # output_quat = output[3:7]

            # rotate output from tip to TCP before sending it to the robot
            T_base_tip_out = pt.transform_from_pq(np.hstack((x_desired, orient_desired)))
            T_base_tcp_out = T_base_tip_out @ T_tip_tcp
            base_tcp_out_ur_pose = get_transform_as_ur_pose_rotvec(T_base_tcp_out)
            # print("base_tcp_out_ur_pose: ", base_tcp_out_ur_pose)

            # set position target of the robot
            ur_robot.servoL(base_tcp_out_ur_pose, vel, acc, dt, lookahead_time, gain)    
            ur_robot.waitPeriod(start_time)
            
            # write data in a file
            if iteration_counter >= 2500:
                data_to_store.append(graph)
                data_to_store.append(robot_pose)
            
            counter = counter + dt
            iteration_counter += 1
            
            end_time = time.time()
            elapsed_time = end_time - start_time_
            # print("Loop executed in", elapsed_time, "seconds")
        
        end_time_T = time.time()
        period = end_time_T - start_time_T
        print("period is: ", period)
        
        if mode == "train":
            batch_write(csv_file_path, data_to_store)
            #Now plot the data
            plot_last_assembly(csv_file_path)
        elif mode == "validation":
            batch_write(csv_validation, data_to_store)
        else:
            print("\n\nThe input mode doesn't correspond to any of the possibilities. Data file not being saved!\n\n")
        
        if mode == "train":
            # plot_last_assembly(csv_file_path)
            result = input("Success?: ")
            with open(csv_file_path, 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(result)

    except KeyboardInterrupt:
        # adm_controller.stop()
        ur_robot.stop_control()
    
    if mode == "validation":
        ut_ai.verification_assembly(csv_validation, model)

          
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
    parser.add_argument('--model', type=str, default="sim", help='Whether to use the model trained on simulation data or real data (sim/real/novo)')
    parser.add_argument('--mode', type=str, default="train", help='Whether to run the program to record data or validate data (train/validation)')
    return parser.parse_args()          
  
def batch_write(output_file, input_list):
    with open(output_file, 'w') as file:
        for i in range(0, len(input_list), 2):
            pair = [input_list[i:i+1], input_list[i+1:i+2]]
            file.write(f'{pair[0]}\t{pair[1]}\n')


if __name__ == "__main__":
    main()
