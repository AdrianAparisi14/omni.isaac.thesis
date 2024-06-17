#launch Isaac Sim before any other imports
#default first two lines in any standalone application
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False}) # we can also run as headless.

from omni.isaac.core import World
from numpy.linalg import norm

# Add imports here
import sys
import os
from omni.isaac.core.prims import XFormPrim, XFormPrimView, RigidPrimView
from omni.isaac.core.utils.xforms import get_world_pose
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.isaac.core.utils.extensions as extensions_utils
from omni.isaac.core import SimulationContext

from omni.sdu.utils.utilities import utils as ut
from omni.isaac.thesis.force_dataset.utils import utils as thesis_utils
from omni.isaac.core.utils import rotations as r
from pytransform3d import rotations as pr
import asyncio
import omni.kit.app
import time
from omni.sdu.utils.utilities import plot as plt
import csv
import json
from datetime import datetime
import numpy as np
from pxr import Sdf

import threading

# These were on physics before
from omni.sdu.utils.utilities.math_util import wrench_trans, get_robot_pose_as_transform, get_pose_as_transform, get_transform_as_pq, \
                                                                get_transform_as_ur_pose, R_to_rotvec, get_transform_as_pose
from scipy.spatial.transform import Rotation as R
import quaternion
from pytransform3d import transformations as pt
import copy

# Imports network classification

from omni.sdu.ai.utilities import utils as ut_ai
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
import ast  # Library to safely evaluate literal expressions from strings
import random


NOVO_NUCLEUS_ASSETS = "omniverse://sdur-nucleus.tek.sdu.dk/Projects/novo/Assets/"
ADRIAN_NUCLEUS_ASSETS = "omniverse://sdur-nucleus.tek.sdu.dk/Users/adrianaparisi/MasterThesis/Assets/"

world = World()
# world.scene.add_default_ground_plane()

# # Test -------
simulation_context = SimulationContext(stage_units_in_meters=1.0)
# need to initialize physics getting any articulation..etc
simulation_context.initialize_physics()

def step_callback(step_size):
    # print("simulate with step: ", step_size)
    return

def render_callback(event):
    # print("update app with step: ", event.payload["dt"])
    return

simulation_context.add_physics_callback("physics_callback", step_callback)
simulation_context.add_render_callback("render_callback", render_callback)
simulation_context.stop()
simulation_context.play()

print("step physics & rendering once with a step size of 1/500 second")
simulation_context.set_simulation_dt(physics_dt=1/120.0, rendering_dt=1/240.0)
# simulation_context.step(render=False)
# # End test -----


# Create scene
class dataset_generation():
    def __init__(self) -> None:
        self.tool = None
        self.T_tcp = None
        
        self.Fg = None
        self.Foff = None
        self.Moff = None
        self.r = None
    
    def create_scene(self, world, tool):
        # Added for the app purpose
        import omni
        from omni.isaac.core.utils.prims import get_prim_at_path

        self.tool = tool
        
        # self.physics_dt = 10
        physx_scene = get_prim_at_path("/physicsScene")
        physx_scene.GetAttribute("physxScene:enableGPUDynamics").Set(True)
        physx_scene.GetAttribute("physics:gravityMagnitude").Set(9.82)
        # physx_scene.GetAttribute("physxScene:timeStepsPerSecond").Set(self.physics_dt)
                
        self._world = world
        # Add a default ground plane to the scene in a specific z position
        self._world.scene.add_default_ground_plane(-0.74)
        
        extensions_utils.disable_extension(extension_name="omni.physx.flatcache")
        
        # Add the table to the stage
        add_reference_to_stage(usd_path= NOVO_NUCLEUS_ASSETS + "Siegmund table/Single_Siegmund_table.usd",prim_path="/World/sigmund_table")
        
        # Add plate for testing
        self.add_plate(prim_path="/World/CranfieldBenchmark",position=[0.0,0.1,0.01], orientation=r.euler_angles_to_quat(euler_angles=[0,0,0],degrees=True))
        
        # Add piece to be inserted for testing
        self.add_piece(prim_path="/World/cranfield_piece",position=[0.2,0.1,0.12], orientation=r.euler_angles_to_quat(euler_angles=[0,180,0],degrees=True))
        
        # Add UR5 robot to the scene
        self.robots = dataset_generation.place_robots(self)
        
        # Add fingertips
        # if self.tool is not None:
        #     add_reference_to_stage(usd_path= ADRIAN_NUCLEUS_ASSETS + self.tool,prim_path="/World/fingertip")
        #     ut.move_prim("/World/fingertip",self.robots._gripper.gripper_prim_path + "/tcp/fingertip")    
            # thesis_utils.attach_fingertip_snap(gripper="crg200")
        
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
        
        # Load the gripper force compensation data
        data_directory = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/script/tool_calibration/sim"
        file_path = data_directory + "/FComp_params.json"

        # Read data from JSON file
        with open(file_path, "r") as json_file:
            data = json.load(json_file)

        self.Fg = np.array(data["Fg"])
        print("\n\n self.Fg: ", self.Fg)
        self.Foff = np.array(data["Foff"])
        self.Moff = np.array(data["Moff"])
        self.r = np.array(data["r"])
        
        
        robot_name = "Robot" #+id 
        self.robot = self._world.scene.get_object(robot_name)
        self.end_effector_offset = np.array([0,0,0.2]) #0.165+0.096])  # 0.1309 to be changed when receiving the new tool
        
        # Move to initial target point (the robot should have already this joint configuration)
        cranfield="CranfieldBenchmark"
        if self.tool == "square_tool_cranfield.usd":
            init_target = XFormPrim("/World/"+ cranfield +"/batch/square_30_left").get_world_pose()[0]
        rot = None
        rot = r.euler_angles_to_quat([180,0,0],degrees=True)
        self.init_target_random = np.add(init_target,[random.uniform(-0.0001, 0.0001),random.uniform(-0.0001, 0.0001),0.32])
        print("self.init_target_random: ", self.init_target_random)
        self.robot.movePose3(ut.get_pose3(self.init_target_random,rot_quat=rot))
        
        # To make sure the robot is in the indicated position before we read the initial poses
        # for i in range(10):
        #     world.step(render=True) # execute one physics step and one rendering step
            
        # pos_init = XFormPrim("/World/Robot/wrist_3_link").get_world_pose()
        # print(pos_init[1])
            
        # Randomize the starting pose/orientation of the tcp to add uncertainty
        # mean_pos = [pos_init[0][0], pos_init[0][1]]  # Mean (center) of the distribution
        # mean_orient = pos_init[1]
        # cov_matrix = [[0.0000005, 0.0000005], [0.0000005, .0000005]]  # Covariance matrix
        # std_dev = np.radians(0.1)

        # # Generate 2D Gaussian points
        # points = thesis_utils.random_point_from_2d_gaussian(mean_pos, cov_matrix)
        # orientation = thesis_utils.random_orientation_quaternion(mean_orient,std_dev)
        # x = points[0]
        # y = points[1]
        # desired_pose = ut.get_pose3([x,y,pos_init[0][2]],rot_quat=orientation)
        
        # # Move to desired pose
        # self.robot.movePose3(desired_pose)
        
        for i in range(100):
            world.step(render=True) # execute one physics step and one rendering step
            
        # Get the initial poses and transformation matrices
        T_tcp = XFormPrim("/World/Robot/wrist_3_link").get_world_pose()
        robot_tcp_pose = (T_tcp[0],pr.matrix_from_quaternion(T_tcp[1]))
        print("T_tcp: ", T_tcp)
        T_w_tcp = get_pose_as_transform(robot_tcp_pose)
        print("T_w_tcp: ", T_w_tcp)
        
        robot_base_pose = XFormPrim("/World/Robot").get_world_pose()
        robot_base_pose = (robot_base_pose[0],pr.matrix_from_quaternion(robot_base_pose[1]))
        self.T_w_b = get_pose_as_transform(robot_base_pose)
        print("T_w_b", self.T_w_b)
        self.T_w_b_inv = np.linalg.inv(self.T_w_b)
        
        T_b_tcp = self.T_w_b_inv@T_w_tcp
        print("Transformation matrix T_b_tcp: ", T_b_tcp)
        
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
        self.counter_prev = 0.0
        self.counter = 0.0
        self.task_done = False
        
        # self.T_base_tip_pos_init = self.T_w_b_inv * hole_circle_30_left.get_world_pose()[0]
        hole_target = self.get_hole_target(self.T_base_tip_pos_init, self.T_base_tip_quat_init, self.counter)
        hole_target_pose = hole_target[0]
        hole_target_rot = hole_target[1]
        
        # self.robot.movePose3(ut.get_pose3(hole_target_pose,rot_quat=hole_target_rot))

        frequency = 500.0  # Hz: on isaac is possible thanks to this harware
        # self.dt = 1 / frequency
        self.dt = 0.015
        
        # Setup plot data
        self.plot = plt.plot_juggler()
        self.wrist_forces = []
        self.save_forces = []
        
        # Setup filter parameters
        # Example parameters
        self.sampling_freq = 500   # Replace with the actual sampling frequency of your force sensor
        self.cutoff_freq = 50      # Replace with the desired cutoff frequency in Hz
        # self.online_filter = thesis_utils.OnlineButterworthFilter(self.cutoff_freq, self.sampling_freq, num_channels=6)
        
        
        
        print("Starting thread to perform the task... ")
        for i in range(100):
            world.step(render=True) # execute one physics step and one rendering step

        return
    
    def add_plate(self, prim_path, position, orientation):
        add_reference_to_stage(usd_path= ADRIAN_NUCLEUS_ASSETS + "cranfield_benchmark_bigger_holes.usd",prim_path=prim_path)
        XFormPrim(prim_path=prim_path,position=position,orientation=orientation)
        return
    
    def add_piece(self, prim_path, position, orientation):
        add_reference_to_stage(usd_path= ADRIAN_NUCLEUS_ASSETS + self.tool,prim_path=prim_path)
        XFormPrim(prim_path=prim_path,position=position,orientation=orientation)
        return
    
    @staticmethod
    def place_robots(self):
        from omni.sdu.utils.grippers.grippers import ROBOTIQ
        from omni.sdu.utils.grippers.grippers import CRG200
        from omni.sdu.utils.robots.ur5e import UR5E
        
        world_path = "/World/"
        attach_flange = True
        self.end_effector_offset = [0,0,0] # 0.165+0.096] 
        
        name = "Robot"
        gripper = CRG200
        
        init_q = np.array([-0.18621972, -1.59032573,  2.17889386,-2.15856047, -1.57020867, -0.18656359])
                
        robot = UR5E(prim_path=world_path+name,name=name,
                                    # usd_path=usd_path,
                                    attact_flange=attach_flange,
                                    end_effector_offset=self.end_effector_offset,
                                    initial_joint_q=init_q,
                                    # initial_joint_q=np.array([0.0,-1.5707,1.5707,-1.5707,-1.5707,0.0]),
                                    position=[0.0,-0.3,0.0],
                                    orientation=r.euler_angles_to_quat(euler_angles=[0,0,90],degrees=True),
                                    gripper=gripper,
                                    tool=self.tool)
        # robot._articulation_view._enable_dof_force_sensors = True
        robot._end_effector_offset = [0,0,0] # 0.165+0.096] 
               
        return robot 
    
    def get_hole_target(self, pose, orientation, timestep):
        #cranfield = "CranfieldBenchmark"
        import copy
        from scipy.spatial.transform import Rotation as R
        
        # Pose
        hole_target_pose = copy.deepcopy(pose)
        hole_target_pose[2] = hole_target_pose[2] - timestep/80 # before timestep/12
        # print("hole_target_pose[2]: ", hole_target_pose[2])
        if hole_target_pose[2] < 0.0385: # Correct assembly: 0.0033 // Stop early: 0.0115 // Stop late: 0.0007
            hole_target_pose[2] = 0.0385
        
        # Orientation: there is not rotation movement here
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
    
    def get_piece_target(self, current_pose, target_pose, timestep):
        #cranfield = "CranfieldBenchmark"
        import copy
        from scipy.spatial.transform import Rotation as R
                
        # Pose
        x_difference = target_pose[0] - current_pose[0]
        y_difference = target_pose[1] - current_pose[1]
        z_difference = target_pose[2] - current_pose[2]
        
        if abs(x_difference) < timestep/40:
            x_desired = target_pose[0]
        elif x_difference < 0.0:
            x_desired = current_pose[0] - timestep/40
        elif x_difference > 0.0:
            x_desired = current_pose[0] + timestep/40
        
        if abs(y_difference) < timestep/40:
            y_desired = target_pose[1]
        elif y_difference < 0.0:
            y_desired = current_pose[1] - timestep/40
        elif y_difference > 0.0:
            y_desired = current_pose[1] + timestep/40
        
        if abs(z_difference) < timestep/40:
            z_desired = target_pose[2]
        elif z_difference < 0.0:
            z_desired = current_pose[2] - timestep/40
        elif z_difference > 0.0:
            z_desired = current_pose[2] + timestep/40
    
        return x_desired, y_desired, z_desired
    
    def force_compensation(self, Fg, Foff, Moff, r, F, R1):
        Fin = F[:3]
        Min = F[3:]

        # Apply transformation to Fin and Min
        Fm = Fin
        Mm = Min
        
        # Make R1 fit
        R1[:, [0, 1]] = R1[:, [1, 0]]
        R1[:, 0] *= -1

        # Compute force compensation
        Fcomp = np.dot(R1.T, Fg)
        Fmeas = Fm - Fcomp - Foff
        # Fmeas[2] += 6.85

        # Compute torques
        Mcomp = np.cross(r, Fcomp)
        Mmeas = Mm - Mcomp - Moff

        return Fmeas, Mmeas
            
    def physics_step_prev(self):
        
        ## Testing pick and place ======
        piece = "cranfield_piece"
        target_pick_no_offset = XFormPrim("/World/"+ piece).get_world_pose()[0]
        print(target_pick_no_offset)
        XFormPrim(prim_path="/World/"+ piece,position=np.add(target_pick_no_offset,[random.uniform(-0.0005, 0.0005),random.uniform(-0.005, 0.005),0.0]),orientation=r.euler_angles_to_quat(euler_angles=[0,180,random.uniform(-5, 5)],degrees=True))
        rot = None
        rot = r.euler_angles_to_quat([180,0,0],degrees=True)
        self.robot.movePose3(ut.get_pose3(np.add(target_pick_no_offset,[0.0,0.0,0.3]),rot_quat=rot))
        print("\n\nut.get_pose3(np.add(target_pick_no_offset,[0.0,0.0,0.3]),rot_quat=rot): ",np.add(target_pick_no_offset,[0.0,0.0,0.3]),XFormPrim("/World/"+ piece).get_world_pose()[1])
        for i in range(100):
            world.step(render=True) # execute one physics step and one rendering step
            
        target_pick = np.add(target_pick_no_offset,[0.0,0.0,0.155])
            
        # init pick and place
        print("Starting pick and place ...")
        T_tip_tcp_start = np.add(target_pick_no_offset,[0.0,0.0,0.3])
        T_tip_tcp_current = np.add(target_pick_no_offset,[0.0,0.0,0.3])
        
        target_reached = False   
        while not target_reached:
            # get hole target
            # T_tip_tcp_current = get_world_pose("/World/Robot/tool0/crg200/tcp")
            x_desired, y_desired, z_desired = self.get_piece_target(T_tip_tcp_current, target_pick, self.dt)
            pose_desired = [x_desired, y_desired, z_desired]

            rot = None
            rot = r.euler_angles_to_quat([180,0,0],degrees=True)
            self.robot.movePose3(ut.get_pose3(pose_desired,rot_quat=rot))
            
            T_tip_tcp_current = pose_desired
            
            if np.all(T_tip_tcp_current == target_pick):
                target_reached = True
            
            world.step(render=True)
            # self.counter_prev = self.counter_prev + self.dt
            
        self.robot.close_gripper()
        for i in range(100):
            world.step(render=True) # execute one physics step and one rendering step
        
        target_reached = False   
        while not target_reached:
            # get hole target
            # T_tip_tcp_current = get_world_pose("/World/Robot/tool0/crg200/tcp")
            x_desired, y_desired, z_desired = self.get_piece_target(T_tip_tcp_current, T_tip_tcp_start, self.dt)
            pose_desired = [x_desired, y_desired, z_desired]

            rot = None
            rot = r.euler_angles_to_quat([180,0,0],degrees=True)
            self.robot.movePose3(ut.get_pose3(pose_desired,rot_quat=rot))
            
            T_tip_tcp_current = pose_desired
            
            T_tcp = XFormPrim("/World/Robot/tool0/crg200/tcp").get_world_pose()
            if np.all(T_tip_tcp_current == T_tip_tcp_start):
                target_reached = True
            
            world.step(render=True)
            # self.counter_prev = self.counter_prev - self.dt
        
        
        for i in range(100):
            world.step(render=True) # execute one physics step and one rendering step
            
            
        target_reached = False   
        while not target_reached:
            # get hole target
            # T_tip_tcp_current = get_world_pose("/World/Robot/tool0/crg200/tcp")
            x_desired, y_desired, z_desired = self.get_piece_target(T_tip_tcp_current, self.init_target_random, self.dt)
            pose_desired = [x_desired, y_desired, z_desired]

            rot = None
            rot = r.euler_angles_to_quat([180,0,0],degrees=True)
            self.robot.movePose3(ut.get_pose3(pose_desired,rot_quat=rot))
            
            T_tip_tcp_current = pose_desired
            
            if np.all(T_tip_tcp_current == self.init_target_random):
                target_reached = True
            
            world.step(render=True)
            # self.counter_prev = self.counter_prev - self.dt
        
        for i in range(100):
            world.step(render=True) # execute one physics step and one rendering step
            
        self.task_done = True
        ## End tensting pick and place =====
    
    
    def physics_step(self):
                
        # ================= CONTROLLER STEP =================
        #tcp
        T_tcp = get_world_pose("/World/Robot/wrist_3_link")
        robot_tcp_pose = (T_tcp[0],pr.matrix_from_quaternion(T_tcp[1]))
        T_w_tcp = get_pose_as_transform(robot_tcp_pose)
        
        T_b_tcp = self.T_w_b_inv@T_w_tcp
        
        # T_base_tcp = copy.deepcopy(T_b_tcp)
        T_base_tcp = T_b_tcp.copy()
        T_base_tip = T_base_tcp @ self.T_tcp_tip

        # get tip in base as position and quaternion
        T_base_tip_pq = pt.pq_from_transform(T_base_tip)
        T_base_tip_pos, T_base_tip_quat = T_base_tip_pq[0:3], T_base_tip_pq[3:7]

        # get current robot force-torque in tcp frame (measured at tcp)
        reading = self.robot.get_force_measurement()
        wrist_forces = reading[10]
        self.wrist_forces = [-wrist_forces[0],-wrist_forces[1],-wrist_forces[2],-wrist_forces[3],-wrist_forces[4],-wrist_forces[5]]

        wrist_forces_ = np.array([-wrist_forces[0],-wrist_forces[1],-wrist_forces[2],-wrist_forces[3],-wrist_forces[4],-wrist_forces[5]])
        # self.plot.publish(wrist_forces_.tolist())
        # self.plot.publish(self.wrist_forces)
        # self.plot.publish(reading[-6].tolist())

        # Forces are given in the articulation frame, rotate forces to TCP frame (necessary just in simulation) 
        f_w, mu_w = self.force_compensation(self.Fg, self.Foff, self.Moff, self.r, self.wrist_forces, np.array(pr.matrix_from_quaternion(T_tcp[1])))
        if f_w[2] < 0:
            f_w *= 0.25
        plot_forces = np.array((f_w + mu_w))
        self.save_forces = [f_w[0], f_w[1], f_w[2], mu_w[0], mu_w[1], mu_w[2]]
        self.plot.publish(plot_forces.tolist())
        # f_w, mu_w = self.wrist_forces[:3], self.wrist_forces[3:]
        
        # use wrench transform to place the force torque in the tip.
        mu_tip, f_tip = wrench_trans(mu_w, f_w, self.T_tcp_tip)

        # rotate forces to base frame
        R_base_tip = T_base_tip[:3, :3]
        f_base_tip = R_base_tip @ f_tip

        # get hole target
        out_traj = self.get_hole_target(self.T_base_tip_pos_init, self.T_base_tip_quat_init, self.counter)
        x_desired, quat_desired = out_traj[0], out_traj[1]

        # rotate output from tip to TCP before sending it to the robot
        T_base_tip_out = pt.transform_from_pq(np.hstack((x_desired, quat_desired)))
        T_base_tcp_out = T_base_tip_out @ self.T_tip_tcp
        self.base_tcp_out_ur_pose = get_transform_as_pose(T_base_tcp_out)
        
        # get current robot pose to store data
        # T_current = XFormPrim("/World/Robot/wrist_3_link").get_world_pose()
        # robot_tcp_pose_current = (T_current[0],pr.matrix_from_quaternion(T_current[1]))
        # self.current_robot_pose  = get_transform_as_pose(get_pose_as_transform(robot_tcp_pose_current))

        # set position target of the robot
        world_tcp_out_pose = self.T_w_b@T_base_tcp_out
        world_tcp_out_pose_pq = pt.pq_from_transform(world_tcp_out_pose)
        desired_pose = ut.get_pose3(world_tcp_out_pose_pq[:3],rot_quat=world_tcp_out_pose_pq[3:])
        
        # Move to desired pose
        self.robot.movePose3(desired_pose)
        
        self.counter = self.counter + self.dt
        return
        
   

dataset = dataset_generation()

# Uncomment the name of the tip to be used: the place position will be changed automatically
tool_tip = "square_tool_cranfield.usd"
# tool_tip = None

# Create the scene
dataset.create_scene(world, tool_tip)

def batch_write(output_file, input_list):
        with open(output_file, 'w') as file:
            for i in range(0, len(input_list), 2):
                pair = [input_list[i:i+1], input_list[i+1:i+2]]
                file.write(f'{pair[0]}\t{pair[1]}\n')
                
    
while simulation_app.is_running():
    try:
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
        csv_file_path = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_simulation/csv_snap_position/data" + str(current_time) + ".csv"
            
        print("Starting physics step...")
        
        while not dataset.task_done:#dataset.counter < 0.9: 
            dataset.physics_step_prev()

            simulation_context.step(render=True)
            

        
        while dataset.counter < 9:#dataset.counter < 0.9: 
            period_start = time.time()
            dataset.physics_step()

            # if dataset.counter>5.05:
            #     dataset.wrist_forces[2] = 0
            data_to_store.append(dataset.save_forces)
            data_to_store.append([0.0,0.0,0.0,0.0,0.0,0.0])
            # data_to_store.append(dataset.current_robot_pose.tolist())
    
            # world.step(render=True) # execute one physics step and one rendering step
            # simulation_context.step(render=True)
            simulation_context.step(render=True)
            
            period_end = time.time()
            period = period_end - period_start
            # print("Period of the loop: ", period)

            # if period < dataset.dt:
            #     time.sleep(dataset.dt-period)
                
        # Just to test
        dataset.robot.open_gripper()
        for i in range(100):
            world.step(render=True) # execute one physics step and one rendering step
            
        rot = None
        rot = r.euler_angles_to_quat([180,0,0],degrees=True)
        dataset.robot.movePose3(ut.get_pose3(dataset.init_target_random,rot_quat=rot))
        for i in range(100):
            world.step(render=True) # execute one physics step and one rendering step
        
        # batch_write(csv_file_path, data_to_store)
        
        # result = input("Succes?: ")
        # with open(csv_file_path, 'a', newline='') as csv_file:
        #     csv_writer = csv.writer(csv_file)
        #     csv_writer.writerow(result)
            
        # csv_file_interpolated = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_simulation/csv_snap_position_interpolated/data" + str(current_time) + ".csv"
        # ref_timeseries_path = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_real_robot_sdu/csv_real_robot_admittance_all_correct/data2024-03-11 13:05:44.303659.csv"
        # thesis_utils.interpolate_timeseries(csv_file_path, ref_timeseries_path, csv_file_interpolated)
        
        # with open(csv_file_interpolated, 'a', newline='') as csv_file:
        #     csv_writer = csv.writer(csv_file)
        #     csv_writer.writerow(result)
                    
        # verification_assembly(csv_file_path)
    except KeyboardInterrupt:
        print("Exit task\n\n")