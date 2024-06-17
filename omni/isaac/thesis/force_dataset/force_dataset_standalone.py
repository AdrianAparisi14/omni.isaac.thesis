#launch Isaac Sim before any other imports
#default first two lines in any standalone application
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False}) # we can also run as headless.

from omni.isaac.core import World
from numpy.linalg import norm

# Add imports here
import sys
import os
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.isaac.core.utils.extensions as extensions_utils
from omni.sdu.utils.utilities import utils as ut
from omni.isaac.core.utils import rotations as r
from pytransform3d import rotations as pr
import asyncio
import omni.kit.app
import time
from omni.sdu.utils.utilities import plot as plt
import csv
from datetime import datetime
import numpy as np
from scipy.signal import butter, lfilter
from scipy.stats import multivariate_normal
from pxr import Sdf

# Imports network classification

from omni.sdu.ai.utilities import utils as ut_ai
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
import ast  # Library to safely evaluate literal expressions from strings


NOVO_NUCLEUS_ASSETS = "omniverse://sdur-nucleus.tek.sdu.dk/Projects/novo/Assets/"
ADRIAN_NUCLEUS_ASSETS = "omniverse://sdur-nucleus.tek.sdu.dk/Users/adrianaparisi/MasterThesis/Assets/"

world = World()
# world.scene.add_default_ground_plane()

class OnlineButterworthFilter:
    def __init__(self, cutoff_freq, sampling_freq, order=4, num_channels=6):
        nyquist = 0.5 * sampling_freq
        normal_cutoff = cutoff_freq / nyquist
        self.b, self.a = butter(order, normal_cutoff, btype='low', analog=False)
        self.num_channels = num_channels
        self.filtered_data = [[] for _ in range(num_channels)]

    def update(self, new_data):
        if len(new_data) != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, but got {len(new_data)}.")

        filtered_value = lfilter(self.b, self.a, new_data, axis=0)
        for i in range(self.num_channels):
            self.filtered_data[i].append(filtered_value[i])

    def get_filtered_data(self):
        return [channel_data[-1] for channel_data in self.filtered_data]


# Create scene
class dataset_generation(OnlineButterworthFilter):
    def __init__(self) -> None:
        self.tool = None
    
    def create_scene(self, world, tool):
        # Added for the app purpose
        import omni
        from omni.isaac.core.utils.prims import get_prim_at_path

        self.tool = tool
        
        self.physics_dt = 500
        physx_scene = get_prim_at_path("/physicsScene")
        physx_scene.GetAttribute("physxScene:enableGPUDynamics").Set(True)
        physx_scene.GetAttribute("physics:gravityMagnitude").Set(9.82)
        physx_scene.GetAttribute("physxScene:timeStepsPerSecond").Set(self.physics_dt)
                
        self._world = world
        # Add a default ground plane to the scene in a specific z position
        self._world.scene.add_default_ground_plane(-0.74)
        
        extensions_utils.disable_extension(extension_name="omni.physx.flatcache")
        
        # Add the table to the stage
        add_reference_to_stage(usd_path= NOVO_NUCLEUS_ASSETS + "Siegmund table/Single_Siegmund_table.usd",prim_path="/World/sigmund_table")
        
        # Add plate for testing
        self.add_plate(prim_path="/World/CranfieldBenchmark",position=[0.0,0.1,0.0], orientation=r.euler_angles_to_quat(euler_angles=[0,0,0],degrees=True))
        
        # Add UR5 robot to the scene
        self.robots = dataset_generation.place_robots(self)
        
        # Add fingertips
        add_reference_to_stage(usd_path= ADRIAN_NUCLEUS_ASSETS + self.tool,prim_path="/World/fingertip")
        # usd_path_test = "omniverse://sdur-nucleus.tek.sdu.dk/Users/adrianaparisi/MasterThesis/Assets/snap_tool_001_flat.usd"
        # add_reference_to_stage(usd_path= usd_path_test,prim_path="/World/fingertip")
        # ut.move_prim("/World/fingertip",self.robots._gripper.gripper_prim_path + "/add_tcp/fingertip")        
        
        # omni.kit.commands.execute('RemoveRelationshipTarget',
        #     relationship=get_prim_at_path("/World/Robot/tool0/hand_e/tcp/FixedJoint").GetRelationship('physics:body1'),
        #     target=Sdf.Path("/World/Robot/tool0/hand_e/tcp"))
        # omni.kit.commands.execute('AddRelationshipTarget',
        #     relationship=get_prim_at_path("/World/Robot/tool0/hand_e/tcp/FixedJoint").GetRelationship('physics:body1'),
        #     target=Sdf.Path("/World/Robot/tool0/hand_e/add_tcp/fingertip/hook_body"))
        
        # self.wrench = ArticulationView(prim_paths_expr="/World/Robots/robotA/flange")
        self._world.scene.add(self.robots)
        
        #for the later trajectory circle
        self.counter = 0.0
        self.trigger_admittance = False
         
        return
    
    def init_step(self):
        
        # Imports regarding Admittance controller
        import time
        import quaternion
        from pytransform3d import transformations as pt
        from pytransform3d import rotations as pr
        from pytransform3d.transform_manager import TransformManager

        from scipy.spatial.transform import Rotation as R

        from omni.sdu.utils.utilities.math_util import wrench_trans, get_robot_pose_as_transform, get_pose_as_transform, get_transform_as_pq, \
                                                                get_transform_as_ur_pose, R_to_rotvec, get_transform_as_pose
        from omni.sdu.utils.utilities.admittance_controller_position import AdmittanceControllerPosition
        
        # await omni.kit.app.get_app().next_update_async()
        
        robot_name = "Robot" #+id 
        self.robot = self._world.scene.get_object(robot_name)
        self.end_effector_offset = np.array([0,0,0.2]) #0.165+0.096])  # 0.1309 to be changed when receiving the new tool
        # self.trigger = True
        
        # Move to initial target point (the robot should have already this joint configuration)
        cranfield="CranfieldBenchmark"
        if self.tool == "tool_circle_30mm_flat.usd":
            init_target = XFormPrim("/World/"+ cranfield +"/batch/circle_30_left").get_world_pose()[0]
        elif self.tool == "tool_circle_40mm_flat.usd":
            init_target = XFormPrim("/World/"+ cranfield +"/batch/circle_40_center").get_world_pose()[0]
        elif self.tool == "tool_circle_42mm_02.usd":
            init_target = XFormPrim("/World/"+ cranfield +"/batch/circle_40_center").get_world_pose()[0]
        elif self.tool == "tool_square_30mm_flat.usd":
            init_target = XFormPrim("/World/"+ cranfield +"/batch/square_30_left").get_world_pose()[0]
        elif self.tool == "snap_tool_001_flat.usd":
            init_target = XFormPrim("/World/"+ cranfield +"/batch/circle_40_center").get_world_pose()[0]
        rot = None
        rot = r.euler_angles_to_quat([180,0,0],degrees=True)
        self.robot.movePose3(ut.get_pose3(np.add(init_target,[0.0,0.0,0.27]),rot_quat=rot))
        
        # To make sure the robot is in the indicated position before we read the initial poses
        for i in range(10):
            world.step(render=True) # execute one physics step and one rendering step
            
        pos_init = XFormPrim("/World/Robot/wrist_3_link").get_world_pose()
        print(pos_init[1])
            
        mean_pos = [pos_init[0][0], pos_init[0][1]]  # Mean (center) of the distribution
        mean_orient = pos_init[1]
        cov_matrix = [[0.000001, 0.000001], [0.000001, 0.000001]]  # Covariance matrix
        std_dev = np.radians(0.5)

        # Generate 2D Gaussian points
        points = self.random_point_from_2d_gaussian(mean_pos, cov_matrix)
        orientation = self.random_orientation_quaternion(mean_orient,std_dev)
        x = points[0]
        y = points[1]
        desired_pose = ut.get_pose3([x,y,pos_init[0][2]],rot_quat=orientation)
        
        # Move to desired pose
        self.robot.movePose3(desired_pose)
        
        for i in range(10):
            world.step(render=True) # execute one physics step and one rendering step

        # Get the initial poses and transformation matrices
        T_tcp = XFormPrim("/World/Robot/wrist_3_link").get_world_pose()
        # quat = R.from_quat(T_tcp[1])
        robot_tcp_pose = (T_tcp[0],pr.matrix_from_quaternion(T_tcp[1]))
        print("T_tcp: ", T_tcp)
        T_w_tcp = get_pose_as_transform(robot_tcp_pose)
        print("T_w_tcp: ", T_w_tcp)
        
        robot_base_pose = XFormPrim("/World/Robot").get_world_pose()
        # quat = R.from_quat(robot_base_pose[1])
        robot_base_pose = (robot_base_pose[0],pr.matrix_from_quaternion(robot_base_pose[1]))
        self.T_w_b = get_pose_as_transform(robot_base_pose)
        print("T_w_b", self.T_w_b)
        self.T_w_b_inv = np.linalg.inv(self.T_w_b)
        
        T_b_tcp = self.T_w_b_inv@T_w_tcp
        print("Transformation matrix T_b_tcp: ", T_b_tcp)
        
        # ======= Setup Admittance parameters =========
        # get current robot pose
        T_base_tcp = T_b_tcp

        # define a tip wrt. the TCP frame: THIS IS TECHNICALLY NOT NEEDED AS LONG AS THE OFFSET FOR THE TCP IS WELL DEFINED AT THE BEGGINING OF THE PROGRAM
        quat_identity = quaternion.as_float_array(quaternion.quaternion(1.0, 0.0, 0.0, 0.0))
        self.T_tcp_tip = pt.transform_from_pq(np.hstack((np.array(self.end_effector_offset ), quat_identity)))
        self.T_tip_tcp = np.linalg.inv(self.T_tcp_tip)
        T_base_tip = T_b_tcp @ self.T_tcp_tip
        print("T_base_tip: ", T_base_tip)

        # get tip in base as position and quaternion
        self.T_base_tip_pq = pt.pq_from_transform(T_base_tip)
        print("T_base_tip_pq: ", self.T_base_tip_pq)
        self.T_base_tip_pos_init = self.T_base_tip_pq[0:3]
        self.T_base_tip_quat_init = self.T_base_tip_pq[3:7]
        
        # Cranfield Benchmark
        self.counter = 0.0
        
        # self.T_base_tip_pos_init = self.T_w_b_inv * hole_circle_30_left.get_world_pose()[0]
        hole_target = self.get_hole_target(self.T_base_tip_pos_init, self.T_base_tip_quat_init, self.counter)
        hole_target_pose = hole_target[0]
        hole_target_rot = hole_target[1]
        
        # self.robot.movePose3(ut.get_pose3(hole_target_pose,rot_quat=hole_target_rot))

        # Start the admittance controller
        self.adm_controller = AdmittanceControllerPosition(start_position=hole_target_pose, start_orientation=self.T_base_tip_quat_init)

        # The controller parameters can be changed, the following parameters corresponds to the default,
        # and we set them simply to demonstrate that they can be changed.
        # self.adm_controller.M = np.diag([22.5, 22.5, 22.5])#self.adm_controller.M = np.diag([22.5, 22.5, 22.5])
        # self.adm_controller.D = np.diag([1000, 1000, 1000]) # self.adm_controller.D = np.diag([160, 160, 160])
        # self.adm_controller.K = np.diag([54, 54, 54])# self.adm_controller.K = np.diag([54, 54, 54])

        # self.adm_controller.Mo = np.diag([0.25, 0.25, 0.25])
        # self.adm_controller.Do = np.diag([100, 100, 100])
        # self.adm_controller.Ko = np.diag([10, 10, 10])# self.adm_controller.Ko = np.diag([10, 10, 10])
        
        # The ones copied from the real robot
        self.adm_controller.M = np.diag([22.5,22.5,22.5])# 22.5, 22.5, 22.5])
        self.adm_controller.D = np.diag([3000, 3000, 3000]) # 160, 160, 160])
        self.adm_controller.K = np.diag([20, 20, 20])# [54, 54, 54])

        self.adm_controller.Mo = np.diag([0.25, 0.25, 0.25])
        self.adm_controller.Do = np.diag([200, 200, 200])
        self.adm_controller.Ko = np.diag([7, 7, 7]) # [10, 10, 10])
        
        # Setup plot data
        # self.plot = plt.plot_juggler()
        self.wrist_forces = []
        
        # Setup filter parameters
        # Example parameters
        self.sampling_freq = 500   # Replace with the actual sampling frequency of your force sensor
        self.cutoff_freq = 50      # Replace with the desired cutoff frequency in Hz
        self.online_filter = OnlineButterworthFilter(self.cutoff_freq, self.sampling_freq, num_channels=6)
        
        print("Starting thread to perform the task... ")

        return
    
    def add_plate(self, prim_path, position, orientation):
        add_reference_to_stage(usd_path= ADRIAN_NUCLEUS_ASSETS + "cranfield_benchmark_bigger_holes.usd",prim_path=prim_path)
        XFormPrim(prim_path=prim_path,position=position,orientation=orientation)
        return
    
    def random_point_from_2d_gaussian(self, mean, cov_matrix):
        """
        Generate a random point from a 2D Gaussian distribution.

        Parameters:
        - mean: Mean of the distribution (2D array, e.g., [mean_x, mean_y])
        - cov_matrix: Covariance matrix (2x2 array)

        Returns:
        - Random point sampled from the 2D Gaussian distribution
        """
        point = np.random.multivariate_normal(mean, cov_matrix)
        return point
    
    def random_orientation_quaternion(self, mean_quaternion, std_dev_angle):
        from scipy.spatial.transform import Rotation
        """
        Generate a random orientation (quaternion representation) around a specified mean in 3D space.

        Parameters:
        - mean_quaternion: Mean quaternion (4-element array, e.g., [w, x, y, z])
        - std_dev_angle: Standard deviation of rotation angle in radians

        Returns:
        - Random orientation (quaternion representation) sampled around the specified mean
        """
        random_rotation = Rotation.from_quat(
            np.concatenate([[np.random.normal(0, std_dev_angle)], mean_quaternion[1:]])
        )
        return random_rotation.as_quat()

    
    @staticmethod
    def place_robots(self):
        from omni.sdu.utils.grippers.grippers import ROBOTIQ
        from omni.sdu.utils.robots.ur5e import UR5E
        
        world_path = "/World/"
        attach_flange = True
        self.end_effector_offset = [0,0,0] # 0.165+0.096] 
        
        name = "Robot"
        gripper = ROBOTIQ
        if self.tool == "tool_circle_30mm_flat.usd":
            init_q = np.array([-0.08324718, -1.22508277,  1.73911381, -2.08570248, -1.57088135, -0.08324081]) #with + 0.275
            # init_q = np.array([-0.08324207, -1.22032397,  1.74400288, -2.09510366, -1.57091191, -0.08317883]) # +0.01 with the correct tcp + tip offset
        elif self.tool == "tool_circle_40mm_flat.usd":
            init_q = np.array([-0.26448643, -1.42121788,  1.99589811 ,-2.14580057 ,-1.57058578, -0.26488203])
            # init_q = np.array([-0.26452291 ,-1.41552179 , 2.00113141 ,-2.15625234 ,-1.57040839, -0.26474397]) # +0.01 with the correct tcp + tip offset
        elif self.tool == "tool_circle_42mm_02.usd":
            init_q = np.array([-0.26448643, -1.42121788,  1.99589811 ,-2.14580057 ,-1.57058578, -0.26488203])
        elif self.tool == "tool_square_30mm_flat.usd":
            init_q = np.array([-0.18621972, -1.59032573,  2.17889386,-2.15856047, -1.57020867, -0.18656359])
            # init_q = np.array([-0.18624443, -1.58396523,  2.18528621 ,-2.1722837  ,-1.57010214 ,-0.18688321]) # +0.01 with the correct tcp + tip offset
        elif self.tool == "snap_tool_001_flat.usd":
            init_q = np.array([-0.26448643, -1.42121788,  1.99589811 ,-2.14580057 ,-1.57058578, -0.26488203])
        
        robot = UR5E(prim_path=world_path+name,name=name,
                                    # usd_path=usd_path,
                                    attact_flange=attach_flange,
                                    end_effector_offset=self.end_effector_offset,
                                    initial_joint_q=init_q,
                                    # initial_joint_q=np.array([0.0,-1.5707,1.5707,-1.5707,-1.5707,0.0]),
                                    position=[0.0,-0.3,0.0],
                                    orientation=r.euler_angles_to_quat(euler_angles=[0,0,90],degrees=True),
                                    gripper=gripper)
        # robot._articulation_view._enable_dof_force_sensors = True
        robot._end_effector_offset = [0,0,0] # 0.165+0.096] 
               
        return robot 
    
    def get_hole_target(self, pose, orientation, timestep):
        #cranfield = "CranfieldBenchmark"
        import copy
        from scipy.spatial.transform import Rotation as R
        
        # Pose
        hole_target_pose = copy.deepcopy(pose)
        hole_target_pose[2] = hole_target_pose[2] - timestep/12
        # print("hole_target_pose[2]: ", hole_target_pose[2])
        if hole_target_pose[2] < 0.0051:
            hole_target_pose[2] = 0.0051
        
        # Orientation
        # Original quaternion
        original_quaternion = copy.deepcopy(orientation)

        # Axis of rotation (normalized) and the angle in radians
        axis_of_rotation = np.array([0, 0, 1])
        angle_in_radians = timestep*2

        # Create a quaternion representing the rotation
        rotation_quaternion = pr.quaternion_from_axis_angle(np.concatenate((axis_of_rotation, [angle_in_radians])))

        # Rotate the original quaternion
        rotated_quaternion = pr.concatenate_quaternions(rotation_quaternion, original_quaternion)
    
        return hole_target_pose, orientation
    
    
    def physics_step(self):
        from omni.sdu.utils.utilities.math_util import wrench_trans, get_robot_pose_as_transform, get_pose_as_transform, get_transform_as_pq, \
                                                                get_transform_as_ur_pose, R_to_rotvec, get_transform_as_pose
        from scipy.spatial.transform import Rotation as R
        import quaternion
        from pytransform3d import transformations as pt
        import copy
        
        # ================= ADMITTANCE CONTROLLER STEP =================
        # The admittance step is performed here on the physics thread: make sure the frequency of this thread is 500Hz
        # if self.trigger_admittance == True:
        frequency = 500.0  # Hz: on isaac is possible thanks to this harware
        dt = 1 / frequency
        #tcp
        T_tcp = XFormPrim("/World/Robot/wrist_3_link").get_world_pose()
        robot_tcp_pose = (T_tcp[0],pr.matrix_from_quaternion(T_tcp[1]))
        T_w_tcp = get_pose_as_transform(robot_tcp_pose)
        
        T_b_tcp = self.T_w_b_inv@T_w_tcp
        
        T_base_tcp = copy.deepcopy(T_b_tcp)
        T_base_tip = T_base_tcp @ self.T_tcp_tip

        # get tip in base as position and quaternion
        T_base_tip_pq = pt.pq_from_transform(T_base_tip)
        T_base_tip_pos = T_base_tip_pq[0:3]
        T_base_tip_quat = T_base_tip_pq[3:7]

        # get current robot force-torque in tcp frame (measured at tcp)
        reading = self.robot.get_force_measurement()
        wrist_forces = reading[10]
        # print(reading)
        # print(f"Force: {wrist_forces[:3]}, Torque: {wrist_forces[3:]}")
        self.wrist_forces = [-wrist_forces[0],-wrist_forces[1],-wrist_forces[2],-wrist_forces[3],-wrist_forces[4],-wrist_forces[5]]
        
        #filter data
        self.online_filter.update(self.wrist_forces)
            
        # get filtered data
        filtered_data = self.online_filter.get_filtered_data()

        wrist_forces_ = np.array([-wrist_forces[0],-wrist_forces[1],-wrist_forces[2],-wrist_forces[3],-wrist_forces[4],-wrist_forces[5]])
        filtered_data_ = np.array([filtered_data[0],filtered_data[1],filtered_data[2],filtered_data[3],filtered_data[4],filtered_data[5]])
        # self.plot.publish(wrist_forces_.tolist())
        # self.plot.publish(reading[9].tolist())

        # Forces are given in the articulation frame, rotate forces to TCP frame (necessary just in simulation) 
        f_w = self.wrist_forces[:3]
        mu_w = self.wrist_forces[3:]
        
        # use wrench transform to place the force torque in the tip.
        mu_tip, f_tip = wrench_trans(mu_w, f_w, self.T_tcp_tip)

        # rotate forces to base frame
        R_base_tip = T_base_tip[:3, :3]
        f_base_tip = R_base_tip @ f_tip
        # graph = [f_base_tip[0],f_base_tip[1],f_base_tip[2],mu_tip[0],mu_tip[1],mu_tip[2]] # to plot the forces in base frame
        # self.plot.publish(graph.tolist())

        # the input position and orientation is given as tip in base
        self.adm_controller.pos_input = T_base_tip_pos
        self.adm_controller.rot_input = quaternion.from_float_array(T_base_tip_quat)
        self.adm_controller.q_input = self.robot.get_joint_positions()
        self.adm_controller.ft_input = np.hstack((f_base_tip, mu_tip))

        # get hole target
        out_traj = self.get_hole_target(self.T_base_tip_pos_init, self.T_base_tip_quat_init, self.counter)
        x_desired = out_traj[0]
        quat_desired = out_traj[1]
        # print("x_desired: ", x_desired)
        # print("orient_desired: ", quat_desired)
        self.adm_controller.set_desired_frame(x_desired, quaternion.from_float_array(quat_desired))#quaternion.from_float_array(self.T_base_tip_quat_init))

        # step the execution of the admittance controller
        self.adm_controller.step()
        output = self.adm_controller.get_output()
        output_position = output[0:3]
        output_quat = output[3:7]

        # rotate output from tip to TCP before sending it to the robot
        T_base_tip_out = pt.transform_from_pq(np.hstack((output_position, output_quat)))
        T_base_tcp_out = T_base_tip_out @ self.T_tip_tcp
        self.base_tcp_out_ur_pose = get_transform_as_pose(T_base_tcp_out)
        
        # get current robot pose to store data
        T_current = XFormPrim("/World/Robot/wrist_3_link").get_world_pose()
        robot_tcp_pose_current = (T_current[0],pr.matrix_from_quaternion(T_current[1]))
        self.current_robot_pose  = get_transform_as_pose(get_pose_as_transform(robot_tcp_pose_current))

        # set position target of the robot
        # print("output of the admittance controller: ", pt.pq_from_transform(T_base_tcp_out))
        world_tcp_out_pose = self.T_w_b@T_base_tcp_out
        # print("world_tcp_out_pose: ", world_tcp_out_pose )
        world_tcp_out_pose_pq = pt.pq_from_transform(world_tcp_out_pose)
        desired_pose = ut.get_pose3(world_tcp_out_pose_pq[:3],rot_quat=world_tcp_out_pose_pq[3:])
        
        # Move to desired pose
        self.robot.movePose3(desired_pose)
        
        # # delete this=======
        # cranfield="CranfieldBenchmark"
        # hole_circle_30_left_pose = XFormPrim("/World/"+ cranfield +"/batch/circle_30_left").get_world_pose()[0]
        # hole_circle_40_pose = XFormPrim("/World/"+ cranfield +"/batch/circle_40_center").get_world_pose()[0]
        # hole_square_30_left_pose = XFormPrim("/World/"+ cranfield +"/batch/square_30_left").get_world_pose()[0]
        # rot = r.euler_angles_to_quat([180,0,0],degrees=True)
        # print("inverskin: ", self.robot.get_inverse_kinematics(ut.get_pose3(np.add(hole_square_30_left_pose,[0.0,0.0,0.275]),rot_quat=rot)))
        # # #==========
        
        self.counter = self.counter + dt
        return
        
   

dataset = dataset_generation()

# Uncomment the name of the tip to be used: the place position will be changed automatically
# tool_tip =  "tool_circle_30mm_flat.usd"
# tool_tip = "tool_circle_40mm_flat.usd"
# tool_tip = "tool_square_30mm_flat.usd"
# tool_tip = "tool_circle_42mm_02.usd"
tool_tip = "snap_tool_001_flat.usd"

# Create the scene
dataset.create_scene(world, tool_tip)

def batch_write(output_file, input_list):
        with open(output_file, 'w') as file:
            for i in range(0, len(input_list), 2):
                pair = [input_list[i:i+1], input_list[i+1:i+2]]
                file.write(f'{pair[0]}\t{pair[1]}\n')
                

def verification_assembly(csv_file_path, sequence_length=400):
    
    full_df = ut_ai.get_dataframe_from_directory(csv_file_path)
    
    # Combine 'Forces' and 'Positions' into a single feature column 'Features'
    full_df['Features'] = full_df.apply(lambda row: (row[0], row[1]), axis=1)

    # Assuming 'Features' column contains tuples of Forces and Positions
    full_df['Force'] = full_df['Features'].apply(lambda x: x[0])  # Assuming Forces are at index 0 in the tuple
    full_df['Position'] = full_df['Features'].apply(lambda x: x[1])  # Assuming Positions are at index 1 in the tuple

    # Create separate DataFrames for Force and Position
    force_df = pd.DataFrame(full_df['Force'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
    position_df = pd.DataFrame(full_df['Position'].tolist(), columns=['Position_1', 'Position_2', 'Position_3', 'Orientation_1', 'Orientation_2', 'Orientation_3'])

    # Combine Force and Position DataFrames
    X_numeric = pd.concat([force_df, position_df], axis=1)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)  # The features are transformed to have a mean of 0 and a standard deviation of 1

    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X_scaled).float()
    
    # Reshape for LSTM
    X_reshaped_new, _ = ut_ai.reshape_for_lstm(X_tensor.numpy(), np.zeros(X_tensor.shape[0]), sequence_length)

    # Convert to PyTorch tensor
    X_test_tensor_new = torch.from_numpy(X_reshaped_new).float()

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_test_tensor_new = X_test_tensor_new.to(device)
    
    # Load the trained model
    input_size = X_test_tensor_new.shape[2]
    hidden_size = 32 #16 before
    num_layers = 4 # 1 before
    model = ut_ai.LSTMClassifier(input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load('../../../../../Documents/sdu.extensions.hotfix/exts/omni.sdu.ai/omni/sdu/ai/examples/LSTM/model/modelv6L.pth'))
    model.eval()

    # Get prediction:
    input_tensor = X_test_tensor_new
    prediction = ut_ai.predict_with_model(model, input_tensor, device)
    
    # print("\n\n\nRESULT OF THE ASSEMBLY.... : ", prediction)

    # # Apply the model to get predictions
    # model.eval()
    # with torch.no_grad():
    #     predictions = model(X_test_tensor_new)

    # # Post-process the output (e.g., thresholding for binary classification)
    # threshold = 0.5  # You may need to adjust this based on your problem
    # binary_predictions = (predictions[:, 0] > threshold).int().cpu().numpy()

    # # Now 'binary_predictions' contains the predicted labels for the new data
    
while simulation_app.is_running():
    world.step(render=True)
    if world.is_playing():
        if world.current_time_step_index == 0:
            world.reset()
    # Resetting the world needs to be called before querying anything related to an articulation specifically.
    # Its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
    world.reset()

    # Setup write data on a csv
    data_to_store = []
    batch_size = 0

    # wait for frames to render within the Isaac Sim UI
    world.step(render=True)
    dataset.init_step()
    
    # Path to the csv file to store force-torque data
    current_time = datetime.now()
    csv_file_path = "../../../../../Documents/sdu.extensions.hotfix/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv/data" + str(current_time) + ".csv"
    
    print("Starting physics step...")
    
    while dataset.counter < 0.9:#dataset.counter < 0.8: 
        dataset.physics_step()
        # data = dataset.wrist_forces
        # print(dataset.wrist_forces)
        # data_to_store = np.vstack((data_to_store, np.array([[dataset.wrist_forces], [dataset.current_robot_pose]])))
        data_to_store.append(dataset.wrist_forces)
        data_to_store.append(dataset.current_robot_pose.tolist())
        # print(data_to_store)
        # with open(csv_file_path, 'a', newline='') as csv_file:
        #     csv_writer = csv.writer(csv_file)
        #     csv_writer.writerow(data)

        # we have control over stepping physics and rendering in this workflow
        # things run in sync
        world.step(render=True) # execute one physics step and one rendering step
        
    batch_write(csv_file_path, data_to_store)
    
    result = input("Succes?: ")
    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(result)
                
    # verification_assembly(csv_file_path)