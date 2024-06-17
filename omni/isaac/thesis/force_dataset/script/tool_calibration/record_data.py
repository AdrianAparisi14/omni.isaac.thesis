#launch Isaac Sim before any other imports
#default first two lines in any standalone application
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True}) # we can also run as headless.

import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
from omni.sdu.utils.utilities.math_util import wrench_trans, get_robot_pose_as_transform, get_pose_as_transform

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive



class RobotRTDE(RTDEControl, RTDEReceive):
    def __init__(self, ip: str):
        self.ip = ip
        rtde_frequency = 500.0
        RTDEControl.__init__(self, ip, rtde_frequency)
        RTDEReceive.__init__(self, ip, -1., [])
        

def main():
    
    frequency = 500.0  # Hz
    dt = 1 / frequency
    vel = 0.4
    acc = 0.5
    lookahead_time = 0.1
    gain = 1000
    data_directory = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/script/tool_calibration"


    print("Initializing robot...")
    try:
        ur_robot = RobotRTDE("192.168.1.14")
    except Exception as e:
        print("Failed to connect with the robot...\n " + str(e))
        
    print("Zeroing FT sensor...")
    ur_robot.zeroFtSensor()
    # wait for the first data to be received, before initializing and starting the controller
    time.sleep(0.5)
    
    # Predefined poses the robot will move to in order to perform the calibration:
    Jpos1 = [-3.9413, -1.5726, -1.5737, -0.8672, 0.9046, -4.723]
    Jpos2 = [-3.9437, -1.5875, -1.5739, -0.7885, 2.1742, -4.7221]
    Jpos3 = [-3.9414, -1.6106, -1.5735, -2.1000, 1.5931, -4.7221]
    
    Jposes = [Jpos1, Jpos2, Jpos3]
    
    print("Recording data of the robot in 3 different configurations...")
    print("\nReady to move the robot...")
    
    for i in range(3):
        
        rotations = []
        forces = []
        counter = 0
        
        ur_robot.moveJ(Jposes[i], 0.5, 0.3)
        time.sleep(2)
        
        try:
            while counter<=1: #and robot_pose[2]>POSE_IN_BASE[2]:
                start_time = ur_robot.initPeriod()
                
                # get current robot pose
                robot_pose = ur_robot.getActualTCPPose()
                orient_rotvec = R.from_rotvec(robot_pose[3:])
                robot_tcp_pose = (robot_pose[:3],orient_rotvec.as_matrix())
                rot = orient_rotvec.as_matrix()
                # Make R1 fit
                rot[:, [0, 1]] = rot[:, [1, 0]]
                rot[:, 0] *= -1
                rotations.append(rot.tolist())
                
                T_base_tcp = get_pose_as_transform(robot_tcp_pose)
                
                # get current robot force-torque in base (measured at tcp)
                reading_ = ur_robot.getActualTCPForce()
                reading = [reading_[0],reading_[1],reading_[2],reading_[3],reading_[4],reading_[5]]

                f_base = reading[0:3]
                mu_base = reading[3:6]
                
                # rotate forces from base frame to TCP frame (necessary on a UR robot)
                R_tcp_base = np.linalg.inv(T_base_tcp[:3, :3])
                f_tcp = R_tcp_base @ f_base
                mu_tcp = R_tcp_base @ mu_base
                force_tcp = f_tcp + mu_tcp
        
                forces.append(force_tcp)

                counter = counter + dt
                
                ur_robot.waitPeriod(start_time)
        
        except KeyboardInterrupt:
            ur_robot.stop_control()
                
        # print(rotations)
        
        
        if i == 0:
            ldmn = "Fcalib1"
        elif i == 1:
            ldmn = "Fcalib2"
        elif i == 2:
            ldmn = "Fcalib3"
            
        # Convert arrays to Python lists
        rotations_save = rotations
        forces_save = forces

        # Combine both arrays into a dictionary
        data = {"Rotations": rotations_save, "Forces": forces_save}

        # File path to store JSON data
        file_path = data_directory + "/" + ldmn + ".json"

        # Writing data to the JSON file
        with open(file_path, "w") as json_file:
            json.dump(data, json_file)

        print("Data has been stored in " + data_directory)
        
if __name__ == "__main__":
    main()
