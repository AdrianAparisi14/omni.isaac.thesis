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
simulation_context.set_simulation_dt(physics_dt=1/150.0, rendering_dt=1/150.0)
simulation_context.step(render=True)
# # End test -----


# Create scene
class dataset_generation():
    def __init__(self) -> None:
        self.tool = None
        self.T_tcp = None
    
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
        
        # Add UR5 robot to the scene
        self.robots = dataset_generation.place_robots(self)
        
        # Add fingertips
        # add_reference_to_stage(usd_path= ADRIAN_NUCLEUS_ASSETS + self.tool,prim_path="/World/fingertip")
        # ut.move_prim("/World/fingertip",self.robots._gripper.gripper_prim_path + "/add_tcp/fingertip")    
        # thesis_utils.attach_fingertip_snap(gripper="crg200")
        
        # self.wrench = ArticulationView(prim_paths_expr="/World/Robots/robotA/flange")
        self._world.scene.add(self.robots)
        
        #for the later trajectory circle
        self.counter = 0.0
         
        return
    
    def init_step(self):
        
        # Imports regarding Admittance controller
        import time
        import quaternion
        from pytransform3d import transformations as pt
        from pytransform3d import rotations as pr
        from pytransform3d.transform_manager import TransformManager

        from scipy.spatial.transform import Rotation as R
        
        # await omni.kit.app.get_app().next_update_async()
        
        robot_name = "Robot" #+id 
        self.robot = self._world.scene.get_object(robot_name)
        self.end_effector_offset = np.array([0,0,0.2]) #0.165+0.096])  # 0.1309 to be changed when receiving the new tool
        
        # To make sure the robot is in the indicated position before we read the initial poses
        for i in range(10):
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
        
        # ======= Setup parameters =========
        # get current robot pose
        T_base_tcp = T_b_tcp
        
        # Cranfield Benchmark
        self.counter = 0.0
        
        # The admittance step is performed here on the physics thread: make sure the frequency of this thread is 500Hz
        frequency = 500.0  # Hz: on isaac is possible thanks to this harware
        # self.dt = 1 / frequency
        self.dt = 0.010

        # Setup plot data
        self.plot = plt.plot_juggler()
        self.wrist_forces = []
        
        # Setup filter parameters
        # Example parameters
        self.sampling_freq = 500   # Replace with the actual sampling frequency of your force sensor
        self.cutoff_freq = 50      # Replace with the desired cutoff frequency in Hz
        # self.online_filter = thesis_utils.OnlineButterworthFilter(self.cutoff_freq, self.sampling_freq, num_channels=6)
        
        # Predefine the lists where the rotations and forces will be saved
        self.rotations = []
        self.forces = []
        
        # Predefined poses the robot will move to in order to perform the calibration:
        Jpos1 = [-3.9413, -1.5726, -1.5737, -0.8672, 0.9046, -4.723]
        Jpos2 = [-3.9437, -1.5875, -1.5739, -0.7885, 2.1742, -4.7221]
        Jpos3 = [-3.9414, -1.6106, -1.5735, -2.1000, 1.5931, -4.7221]
        
        self.Jposes = [Jpos1, Jpos2, Jpos3]
        
        print("Recording data of the robot in 3 different configurations...")
        print("\nReady to move the robot...")
        
        
        print("Starting thread to perform the task... ")

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
        if self.tool == "snap_tool_001_flat.usd":
            init_q = np.array([-0.3395461,  -1.65434366,  2.30128079, -2.21752396, -1.57156445, -0.33900283])
        
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
    
    def physics_step(self):
        
        #tcp
        T_tcp = get_world_pose("/World/Robot/wrist_3_link")
        robot_tcp_pose = (T_tcp[0],pr.matrix_from_quaternion(T_tcp[1]))
        T_w_tcp = get_pose_as_transform(robot_tcp_pose)
        
        T_b_tcp = self.T_w_b_inv@T_w_tcp
        
        # get current robot pose
        orient_rotvec = pr.matrix_from_quaternion(T_tcp[1])
        rot = orient_rotvec
        self.rotations.append(rot.tolist())
        
        # get current robot force-torque in base (measured at tcp)
        reading = self.robot.get_force_measurement()
        wrist_forces = reading[10]
        force_reading = [-wrist_forces[0],-wrist_forces[1],-wrist_forces[2],-wrist_forces[3],-wrist_forces[4],-wrist_forces[5]]
        wrist_forces_ = np.array([-wrist_forces[0],-wrist_forces[1],-wrist_forces[2],-wrist_forces[3],-wrist_forces[4],-wrist_forces[5]])
        self.plot.publish(wrist_forces_.tolist())
        f_base = force_reading[0:3]
        mu_base = force_reading[3:6]
        self.forces.append(force_reading)

        self.counter = self.counter + self.dt

        return
        
   

dataset = dataset_generation()

# Uncomment the name of the tip to be used: the place position will be changed automatically
tool_tip = "snap_tool_001_flat.usd"

# Create the scene
dataset.create_scene(world, tool_tip)

data_directory = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/script/tool_calibration/sim"
                
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
    csv_file_path = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_simulation/csv_snap_admittance/data" + str(current_time) + ".csv"
        
    print("Starting physics step...")
    
    for i in range(3):
        
        dataset.rotations = []
        dataset.forces = []
        dataset.counter = 0
        
        dataset.robot.moveJ(dataset.Jposes[i])
        # To make sure the robot is in the indicated position before we read the initial poses
        for k in range(1000):
            world.step(render=True) # execute one physics step and one rendering step
        
        try:
            while dataset.counter<=5: #and robot_pose[2]>POSE_IN_BASE[2]:
                period_start = time.time()
                dataset.physics_step()
                simulation_context.step(render=True)
                
                period_end = time.time()
                period = period_end - period_start
                # print("Period of the loop: ", period)

                if period < dataset.dt:
                    time.sleep(dataset.dt-period)
                
        except KeyboardInterrupt:
            print("Exit task\n\n")
            simulation_app.close() # close Isaac Sim
            
        
        if i == 0:
            ldmn = "Fcalib1"
        elif i == 1:
            ldmn = "Fcalib2"
        elif i == 2:
            ldmn = "Fcalib3"
            
        # Convert arrays to Python lists
        rotations_save = np.array(dataset.rotations)
        forces_save = np.array(dataset.forces)

        # Combine both arrays into a dictionary
        data = {"Rotations": rotations_save.tolist(), "Forces": forces_save.tolist()}

        # File path to store JSON data
        file_path = data_directory + "/" + ldmn + ".json"

        # Writing data to the JSON file
        with open(file_path, "w") as json_file:
            json.dump(data, json_file)

        print("Data has been stored in " + data_directory)
                        

simulation_app.close() # close Isaac Sim

