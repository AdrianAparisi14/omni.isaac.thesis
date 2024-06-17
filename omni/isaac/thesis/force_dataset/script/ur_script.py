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

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

from omni.sdu.utils.utilities.math_util import wrench_trans, get_robot_pose_as_transform, get_pose_as_transform, get_transform_as_pq, \
                                                                get_transform_as_ur_pose, R_to_rotvec, get_transform_as_pose, get_transform_as_ur_pose_rotvec
from omni.sdu.utils.utilities.admittance_controller_position import AdmittanceControllerPosition


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
    if hole_target_pose[2] < 0.0051:
        hole_target_pose[2] = 0.0051
    
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
      args ([str]): command line parameter list
    """
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
        ur_robot = RobotRTDE("192.168.1.11")
    except Exception as e:
        print("Failed to connect with the robot...\n " + str(e))
        
    print("Zeroing FT sensor...")
    ur_robot.zeroFtSensor()
    # wait for the first data to be received, before initializing and starting the controller
    time.sleep(0.5)

    # Move robot to the starting position
    print("Moving robot to the start position...")
    ur_robot.moveJ([0.1833338737487793, -1.4872470659068604, 2.0530546347247522, -2.137980123559469, -1.5681775251971644, 2.2128610610961914])
    # ur_robot.moveL([-0.4338166342180918, -0.21722334966328855, 0.2768509807303988, 3.0574024796186383, -0.7080677813361, -0.0026293921812557504])
    time.sleep(2)
    
    # Retrieve Pose of the robot in the starting position
    robot_pose = ur_robot.getActualTCPPose() # [-0.4338166342180918, -0.21722334966328855, 0.2768509807303988, 3.0574024796186383, -0.7080677813361, -0.0026293921812557504] # harcoded for debugging purposes :D
    print("robot_pose_start: ", robot_pose)
    orient_rotvec = R.from_rotvec(robot_pose[3:])
    robot_tcp_pose = (robot_pose[:3],orient_rotvec.as_matrix())
    print("robot_tcp_pose: ", robot_tcp_pose)
    
    T_base_tcp = get_pose_as_transform(robot_tcp_pose)
    print("T_base_tcp: ", T_base_tcp)

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
    print("T_base_tip_circle: ", T_base_tcp_circle)

    # Use moveL to move to the initial point on the circle.
    print("Moving the robot to the start position of the insertion (should be the same...)")
    #print("ur_robot.moveL(get_transform_as_ur_pose(T_base_tcp_circle)): ", get_transform_as_ur_pose_rotvec(T_base_tcp_circle))
    #ur_robot.servoL(get_transform_as_ur_pose_rotvec(T_base_tcp_circle), vel, acc, dt, lookahead_time, gain)

    print("Initializing AdmittanceController...")
    adm_controller = AdmittanceControllerPosition(start_position=x_desired, start_orientation=T_base_tip_quat_init,
                                                  start_ft=ur_robot.getActualTCPForce())
    print("Starting AdmittanceController!")
    
    # time.sleep(20)

    # The controller parameters can be changed, the following parameters corresponds to the default,
    # and we set them simply to demonstrate that they can be changed.
    adm_controller.M = np.diag([22.5,22.5,22.5])# 22.5, 22.5, 22.5])
    adm_controller.D = np.diag([5000, 5000, 5000]) # 160, 160, 160])
    adm_controller.K = np.diag([20, 20, 20])# [54, 54, 54])

    adm_controller.Mo = np.diag([0.25, 0.25, 0.25])
    adm_controller.Do = np.diag([200, 200, 200])
    adm_controller.Ko = np.diag([7, 7, 7]) # [10, 10, 10])
    
    # Setup plot data
    plot = plt.plot_juggler()
    
    # Setup write data on a csv
    data_to_store = []
    batch_size = 0
    current_time = datetime.now()
    csv_file_path = "../../../../../Documents/sdu.extensions.hotfix/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/script/csv/data" + str(current_time) + ".csv"

    try:
        while counter<=8.5:
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
            graph = [f_tcp[0],f_tcp[1],f_tcp[2],mu_tcp[0],mu_tcp[1],mu_tcp[2]]
            plot.publish(graph)

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
            # orient_desired_ = R.from_quat(orient_desired)
            # print("Desired pos orient: ", x_desired, orient_desired_.as_euler('xyz'))
            # T_base_tip_desired = pt.transform_from_pq(np.hstack((x_desired, orient_desired)))
            # print("T_base_tip_desired: ", T_base_tip_desired)
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
            data_to_store = np.hstack((data_to_store, reading))
            # data = reading
            batch_size += 1
            
            counter = counter + dt
        
        batch_write(csv_file_path, data_to_store, batch_size=1)

    except KeyboardInterrupt:
        adm_controller.stop()
        ur_robot.stop_control()


def batch_write(file_path, data, batch_size=1):
    with open(file_path, 'w') as file:
        for i in range(0, len(data), batch_size):
            batch = data[i]
            file.writelines(str(batch) + '\n')


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    main()
