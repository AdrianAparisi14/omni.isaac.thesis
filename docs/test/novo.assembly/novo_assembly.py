from pose_database import PoseDatabase
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_io import RTDEIOInterface as RTDEIo
from robotiq_gripper import RobotiqGripper
import time
import numpy as np
from scipy.spatial.transform import Rotation as R 
import threading
import plot as plt
import argparse
import sys

SHOULD_RUN = False

class RobotRTDE(RTDEControl, RTDEReceive, RTDEIo):
    def __init__(self, ip: str):
        self.ip = ip
        RTDEControl.__init__(self, ip, flags=RTDEControl.FLAG_VERBOSE | RTDEControl.FLAG_UPLOAD_SCRIPT)
        RTDEReceive.__init__(self, ip, -1., [])
        RTDEIo.__init__(self, ip)
        
def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Record data example")
    parser.add_argument(
        "-ip",
        "--robot_ip",
        dest="ip",
        help="IP address of the UR robot",
        type=str,
        default='192.168.1.11',
        metavar="<IP address of the UR robot>")
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="data output (.csv) file to write to (default is \"robot_data.csv\"",
        type=str,
        default="robot_data.csv",
        metavar="<data output file>")
    parser.add_argument(
        "-f",
        "--frequency",
        dest="frequency",
        help="the frequency at which the data is recorded (default is 500Hz)",
        type=float,
        default=500.0,
        metavar="<frequency>")

    return parser.parse_args(args)

def main(args):
        
    # Init database
    host = 'localhost'
    pose_db = PoseDatabase(hostname=host)
    
    # Init robot
    robot_ip = '192.168.1.11'
    robot_rtde = RobotRTDE(ip=robot_ip)
    robot_rtde.setStandardDigitalOut(0, True)
    robot_rtde.zeroFtSensor()
    
    # Data recording
    record_variables = ["timestamp", "actual_TCP_force"]
    robot_rtde.startFileRecording("../../../../../Documents/sdu.extensions/exts/omni.isaac.thesis/docs/test/novo.assembly/robot_data.csv", record_variables)
    # rtde_r = RTDEReceive(args.ip, args.frequency)
    # rtde_r.startFileRecording(args.output)
    print("Data recording started, press [Ctrl-C] to end recording.")
    
    # Start plot publish
    plot = plt.plot_juggler()
    thread1 = force_reading(plot, robot_rtde)

    # Start the thread
    thread1.start()
    
    try:
        # Init gripper
        ip_gripper = '192.168.1.11'
        port_gripper = 63352
        gripper = RobotiqGripper()
        gripper.connect(ip_gripper, port_gripper)
        gripper.activate()
        
        # Exchange fingers
        clear_q_pose_name = 'robotA_finger1_clear_q'
        clear_q_pose = pose_db.get_pose_from_db(clear_q_pose_name)
        attach_pose_name = 'robotA_finger1_attach_pose'
        attach_pose = pose_db.get_pose_from_db(attach_pose_name)
        clear_pose_name = 'robotA_finger1_clear_pose'
        clear_pose = pose_db.get_pose_from_db(clear_pose_name )
        release_pose_name = 'robotA_finger1_release_pose'
        release_pose= pose_db.get_pose_from_db(release_pose_name )
        exchange_fingers(robot_rtde, gripper, clear_q_pose, clear_pose, attach_pose, release_pose)
        
        # Move to the top of the scaledrum
        pose_name = 'robotA_scaledrum_pick_approx_q'
        pose = pose_db.get_pose_from_db(pose_name)
        robot_rtde.moveJ(pose)
        
        # Pick part
        cartesian_pick_approx_name = 'robotA_scaledrum_pick_approx_pose'
        cartesian_pick_approx = pose_db.get_pose_from_db(cartesian_pick_approx_name)
        cartesian_pick_pose_name = 'robotA_scaledurm_pick_pose'
        cartesian_pick_pose = pose_db.get_pose_from_db(cartesian_pick_pose_name)
        no_part_pos = 1
        grip_force = 10.0
        grip_pos = 1.0
        pick_part(robot_rtde, gripper, cartesian_pick_approx, no_part_pos, grip_force, cartesian_pick_pose, grip_pos)
        
        # Move to the top of the housing
        pose_name = 'robotA_scaledrum_insertion_clear_pose'
        pose = pose_db.get_pose_from_db(pose_name)
        robot_rtde.moveL(pose)
        
        # Move to insertion of the scaledrum
        pose_name = 'robotA_scaledrum_insertion_pose'
        pose = pose_db.get_pose_from_db(pose_name)
        robot_rtde.moveL(pose)
        
        # Insert Scaledrum
        insert_scaledrum(robot_rtde, gripper, 5, -2)
        
        # Move to the top of the housing
        pose_name = 'robotA_scaledrum_insertion_clear_pose'
        pose = pose_db.get_pose_from_db(pose_name)
        robot_rtde.moveL(pose)
    except KeyboardInterrupt:
        thread1.stop_flag.set()
        thread1.join()
        robot_rtde.stop_control()
        robot_rtde.stopFileRecording()
        print("\nData recording stopped.")
    
    
    
def exchange_fingers(robot_rtde, gripper,clear_q_pose, clear_pose, attach_pose, release_pose, attach = True):
    gripperPosition = 24
    fingerDigitalOutputNumber = 0
    # Exchange fingers using robotiq gripper
    gripper_normalized = int(gripperPosition * 255 / 100)
    # move gripper to gripperPosition
    gripper.move_and_wait_for_pos(gripper_normalized, 128, 128)
    #time.sleep(0.5)

    # move to the clear q joint pose
    robot_rtde.moveJ(clear_q_pose)

    if attach:
        # move to the attach pose
        robot_rtde.moveL(attach_pose, 0.05, 0.05)
        # attach fingers with pneumatic lock
        robot_rtde.setStandardDigitalOut(fingerDigitalOutputNumber, False)
        # await locking mechanism
        time.sleep(1)
        # move robot to clear pose
        robot_rtde.moveL(clear_pose, 0.05, 0.05)
    else:
        # move to the release pose
        robot_rtde.moveL(release_pose, 0.05, 0.05)
        # release fingers with pneumatics
        robot_rtde.setStandardDigitalOut(fingerDigitalOutputNumber, True)
        time.sleep(1)
        # move robot to clear pose
        robot_rtde.moveL(clear_pose, 0.05, 0.05)
        robot_rtde.moveL(clear_pose, 0.05, 0.05)
        
def pick_part(robot_rtde, gripper, cartesian_pick_approx, no_part_pos, grip_force, cartesian_pick_pose, grip_pos):
    
    robot_rtde.moveL(cartesian_pick_approx)   
    gripper_normalized = int(no_part_pos * 255 / 100)
    force_normalized = int(grip_force * 255 / 100)
    
    # move gripper to gripperPosition
    gripper.move_and_wait_for_pos(gripper.get_closed_position(), 128, force_normalized)

    robot_rtde.moveL(cartesian_pick_pose)
    #db.log_info("Done moving "+db.inputs.urRobot)
    
    gripper_normalized = int(grip_pos * 255 / 100)
    gripper.move_and_wait_for_pos(gripper_normalized, 128, force_normalized)
    time.sleep(0.5)

    robot_rtde.moveL(cartesian_pick_approx)
    
def insert_scaledrum(robot_rtde, gripper, insertionTorque, insertionForce):
    # ===== Move 6th joint to an intermediate value =====
    print("Moving 6th joint to a intermediate value...")
    pose = robot_rtde.getActualQ()
    init_cartesian = robot_rtde.getActualTCPPose()
    init_q = robot_rtde.getActualQ()
    # Task parameters:
    if insertionTorque > 0:
        RZ = 1.5*np.pi #np.pi/2
    else:
        RZ = -np.pi/2

    pose[5] = RZ
    task_frame = [0, 0, 0, 0, 0, 0] # Base-TCP force c.s. transform
    frame_transform_type = 2 # read API
    compliance_vector = [0, 0, 1, 0, 0, 1] # Set compliant axis
    speed = [0, 0, -0.01, 0, 0, 0]    
    F_insert = np.zeros(6)
    F_insert[2] = insertionForce  #[0, 0, -insertionForce] # Virtual force in base frame
    F_insert[5] = insertionTorque
    vMax = [0.1, 0.1, 2, 0.1, 0.1, 2] # Maximum resulting velocity [m/s, m/s, m/s, rad/s, rad/s, rad/s]
    max_allowed_rotation = 2.5 * np.pi
    screwInRotation = np.pi

    time.sleep(0.3)
    robot_rtde.zeroFtSensor() # reset force sensor due to drift
    time.sleep(0.3)

    robot_rtde.moveJ(pose, 2, 2)
    robot_rtde.moveUntilContact(speed)
    # ===== Small insert force to assure contact =====
    try:
        print("Initialization small insert force to assure contact...")
        # Force mode parameters
        searchingForInsert = True

        tcp_pose_initial = robot_rtde.getActualTCPPose()
        initial_q = robot_rtde.getActualQ()
        
        #Rotate with insertionTorque while pressing with insertionForce
        i = 0
        while(searchingForInsert):
            t_start = robot_rtde.initPeriod()

            current_q = robot_rtde.getActualQ()
            current_tcp_pose = robot_rtde.getActualTCPPose()

            robot_rtde.forceMode(task_frame, compliance_vector, F_insert.tolist(), frame_transform_type, vMax)
            if current_tcp_pose[2] - tcp_pose_initial[2] < -0.003:
                searchingForInsert = False
                
            elif abs(current_q[5] - initial_q[5]) >= max_allowed_rotation or abs(current_q[5]) > 1.75 * np.pi:
                robot_rtde.forceModeStop()
                print("TCP OF ROBOT EXCEEDED THE MAXIMUM ROTATION")
                searchingForInsert = False
            
            robot_rtde.waitPeriod(t_start)

        # ==== AFTER FINDING INSERTION, CONTINUE SCREEWING TO ENSURE INSERTION ====
        print("SCREEWING A BIT MORE TO ENSURE INSERTION IN THE ROBOT ")
        initial_q = robot_rtde.getActualQ()
        screwingIn = True
        compliance_vector = [0, 0, 1, 0, 0, 0]
        F_insert[5] = 0
        robot_rtde.forceMode(task_frame, compliance_vector, F_insert.tolist(), frame_transform_type, vMax)
        
        current_q = robot_rtde.getActualQ()
        moveQ = current_q
        moveQ[5] = -1.5 * np.pi
        robot_rtde.moveJ(moveQ, 0.75, 1, asynchronous=True) # ensure screewing is done by force mode in Z and moveJ for rotation of the tcp
        
        while screwingIn:
            current_q = robot_rtde.getActualQ()
            # print("current_q: ", current_q)
            # print("initial_q: ", initial_q)
            # print("diff: ", current_q[5] - initial_q[5])
            # print("screwInRotation: ", screwInRotation)
            if abs(current_q[5] - initial_q[5]) >= screwInRotation or abs(current_q[5]) > 1.45 * np.pi:
                robot_rtde.stopJ(1.0, False)
                print("Stop continue screwing")
                screwingIn = False
            time.sleep(0.1)
        
        # stop force mode and unwind joint 6
        robot_rtde.forceModeStop()
        # Release scaledrum
        gripper.move_and_wait_for_pos(gripper.get_closed_position(), 255, 1)
        time.sleep(1)
        # move linearly to a slightly higher position and come back to the init pose
        higher_pose = robot_rtde.getActualTCPPose()
        higher_pose[2] = init_cartesian[2]
        robot_rtde.moveL(higher_pose, 1, 1)
        robot_rtde.moveJ(init_q, 1, 1)
        
    except Exception as e:
        print(e)
    
def Z_touchup(robot_rtde, speed, expected_difference, tolerance):
    time.sleep(0.2)
    tcp_pose_initial = robot_rtde.getActualTCPPose()
    robot_rtde.zeroFtSensor() # reset force sensor due to drift
    time.sleep(0.2)
    robot_rtde.moveUntilContact(speed)
    current_tcp_pose = robot_rtde.getActualTCPPose()
    deviation = current_tcp_pose[2] - tcp_pose_initial[2]
    print(deviation)
    within_spec = False
    if deviation > expected_difference - tolerance and deviation < expected_difference + tolerance:
        within_spec = True
    
    robot_rtde.moveL(tcp_pose_initial)
    
    return within_spec


class force_reading(threading.Thread):
    def __init__(self, plot, robot_rtde):
        super(force_reading, self).__init__()
        self.stop_flag = threading.Event()
        self.plot = plot
        self.robot_rtde = robot_rtde

    def run(self):
        while not self.stop_flag.is_set():
            # f_base = robot_rtde.ft[0:3]
            # mu_base = robot_rtde.ft[3:6]
            wrist_forces = self.robot_rtde.getActualTCPForce()
            # getActualTCPForce
            
            self.plot.publish(wrist_forces)
        print("Thread is stopping gracefully.")



if __name__ == "__main__":
    # This block will only be executed if the script is run directly, not imported
    main(sys.argv[1:])
    