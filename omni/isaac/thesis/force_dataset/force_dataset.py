from omni.isaac.thesis.base_sample import BaseSample
# This extension has franka related tasks and controllers as well
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
from pxr import Usd
from omni.sdu.utils.utilities import plot as plt
import csv
from datetime import datetime
import numpy as np
import copy
from pxr import Usd, Sdf

import omni
from omni.isaac.core.utils.prims import get_prim_at_path

import time
import quaternion
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
from pytransform3d.transform_manager import TransformManager

from scipy.spatial.transform import Rotation as R

from omni.sdu.utils.utilities.math_util import wrench_trans, get_robot_pose_as_transform, get_pose_as_transform, get_transform_as_pq, \
                                                        get_transform_as_ur_pose, R_to_rotvec, get_transform_as_pose
from omni.sdu.utils.utilities.admittance_controller_position import AdmittanceControllerPosition
        
from omni.sdu.utils.grippers.grippers import ROBOTIQ
from omni.sdu.utils.robots.ur5e import UR5E

from omni.sdu.utils.tasks.tasks import Tasks

from omni.isaac.thesis.force_dataset.utils import utils as thesis_utils

# from sdu_controllers import __version__
# from sdu_controllers.robots.ur_robot import URRobot
from omni.sdu.utils.utilities.math_util import wrench_trans, get_robot_pose_as_transform, get_pose_as_transform, get_transform_as_pq, \
                                                         get_transform_as_ur_pose, R_to_rotvec, get_transform_as_pose
from omni.sdu.utils.utilities.admittance_controller_position import AdmittanceControllerPosition

import json

NOVO_NUCLEUS_ASSETS = "omniverse://sdur-nucleus.tek.sdu.dk/Projects/novo/Assets/"
ADRIAN_NUCLEUS_ASSETS = "omniverse://sdur-nucleus.tek.sdu.dk/Users/adrianaparisi/MasterThesis/Assets/"
CALIBRATION_FILES = os.path.dirname(os.path.realpath(__file__)) + "/calibration_data/"


class SceneForceDataset(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.trigger = False
        self.trigger_admittance = False
        self.robot = None
        return

    def setup_scene(self):
        world = self.get_world()
        self.meters_per_unit = 1.00
        self.physics_dt = 500
        
        physx_scene = get_prim_at_path("/physicsScene")
        physx_scene.GetAttribute("physxScene:enableGPUDynamics").Set(True)
        physx_scene.GetAttribute("physics:gravityMagnitude").Set(9.82)
        physx_scene.GetAttribute("physxScene:timeStepsPerSecond").Set(self.physics_dt)
        
        # Add a default ground plane to the scene in a specific z position
        world.scene.add_default_ground_plane(-0.74)
        
        extensions_utils.disable_extension(extension_name="omni.physx.flatcache")
        
        # Uncomment the tool to be used: the place position will be changed automatically
        # tool_tip =  "tool_circle_30mm_flat.usd"
        # tool_tip = "tool_circle_40mm_flat.usd"
        # tool_tip = "tool_square_30mm_flat.usd"
        # tool_tip = "tool_circle_42mm_02.usd"
        tool_tip = "snap_tool_001_flat.usd"

        self.tool = tool_tip
        
        # Add the table to the stage
        add_reference_to_stage(usd_path= NOVO_NUCLEUS_ASSETS + "Siegmund table/Single_Siegmund_table.usd",prim_path="/World/sigmund_table")
        
        # Add plate for testing
        self.add_plate(prim_path="/World/CranfieldBenchmark",position=[0.0,0.1,0.0], orientation=r.euler_angles_to_quat(euler_angles=[0,0,0],degrees=True))
        
        # Add UR5 robot to the scene
        self.robots = SceneForceDataset.place_robots(self)
        
        # Add fingertips
        add_reference_to_stage(usd_path= ADRIAN_NUCLEUS_ASSETS + self.tool,prim_path="/World/fingertip")
        ut.move_prim("/World/fingertip",self.robots._gripper.gripper_prim_path + "/add_tcp/fingertip")    
        if tool_tip == "snap_tool_001_flat.usd":
            thesis_utils.attach_fingertip_snap()
        
        # self.wrench = ArticulationView(prim_paths_expr="/World/Robots/robotA/flange")
        world.scene.add(self.robots)
        
        #for the later trajectory circle
        self.counter = 0.0
        self.trigger_admittance = False
         
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        self.tasks = None
        self.all_tasks = [] 
        await self._world.play_async()
        return
    
    def get_hole_target(self, pose, orientation, timestep):
        #cranfield = "CranfieldBenchmark"
        import copy
        from scipy.spatial.transform import Rotation as R
        
        # Pose
        hole_target_pose = copy.deepcopy(pose)
        hole_target_pose[2] = hole_target_pose[2] - timestep/20
        
        # Orientation
        # Original quaternion
        original_quaternion = copy.deepcopy(orientation)

        # Axis of rotation (normalized) and the angle in radians
        axis_of_rotation = np.array([0, 0, 1])
        angle_in_radians = timestep

        # Create a quaternion representing the rotation
        rotation_quaternion = pr.quaternion_from_axis_angle(np.concatenate((axis_of_rotation, [angle_in_radians])))

        # Rotate the original quaternion
        rotated_quaternion = pr.concatenate_quaternions(rotation_quaternion, original_quaternion)
    
        return hole_target_pose, rotated_quaternion
    
    def task2(self):
        world = self.get_world()
        robot_name = "Robot" #+id 
        self.robot = self._world.scene.get_object(robot_name)
        self.end_effector_offset = np.array([0,0,0.18]) #0.165+0.096])  # 0.1309 to be changed when receiving the new tool
        # self.trigger = True
        task = Tasks(self.robot)
        
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
        self.robot.movePose3(ut.get_pose3(np.add(init_target,[0.0,0.0,0.275]),rot_quat=rot))
        
        # To make sure the robot is in the indicated position before we read the initial poses
        for i in range(10):
            world.step(render=True) # execute one physics step and one rendering step

        # Get the initial poses and transformation matrices
        # T_tcp = XFormPrim("/World/Robot/wrist_3_link").get_world_pose()
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
        self.adm_controller.M = np.diag([22.5, 22.5, 22.5])#self.adm_controller.M = np.diag([22.5, 22.5, 22.5])
        self.adm_controller.D = np.diag([1000, 1000, 1000]) # self.adm_controller.D = np.diag([160, 160, 160])
        self.adm_controller.K = np.diag([54, 54, 54])# self.adm_controller.K = np.diag([54, 54, 54])

        self.adm_controller.Mo = np.diag([0.25, 0.25, 0.25])
        self.adm_controller.Do = np.diag([100, 100, 100])
        self.adm_controller.Ko = np.diag([10, 10, 10])# self.adm_controller.Ko = np.diag([10, 10, 10])
        
        # Setup plot data
        self.plot = plt.plot_juggler()
        
        print("Starting thread to perform the task... ")
        
        self.trigger_admittance = True
        
        return task

    async def setup_pre_reset(self):
        return

    # This function is called after Reset button is pressed
    # Resetting anything in the world should happen here
    async def setup_post_reset(self):
        self.count = 0
        self.trigger = False
        self.trigger_admittance = False
        await self._world.play_async()

        self.all_tasks = []

        self.robot = None
        self.robots = []
        self.physics_dt = 500
        self.robots = self._world.scene.get_object("Robot")
        
        world_pose = self.robots.get_world_pose()
        self.robots._kinematics.set_robot_base_pose(robot_position=world_pose[0],robot_orientation=world_pose[1])

        # for idx, value in enumerate(['A','B','C','D']):
        #     if value == 'A':
        task = self.task2()
        #     elif value == 'D':
        #         task = self.task2fasterExhange(value)
        #     elif value == 'B':
        #         task = self.taskcolabB(value)
        #     else:
        #         task = self.taskcolabC(value)
        self.all_tasks.append(task)
        
        return

    def world_cleanup(self):
        return
    
    def physics_step(self, step_size):
        self.count += 1
        if len(self.all_tasks) > 0:
            for task in self.all_tasks:
                task.next_step()
        
        # ================= ADMITTANCE CONTROLLER STEP =================
        # The admittance step is performed here on the physics thread: make sure the frequency of this thread is 500Hz
        if self.trigger_admittance == True:
            # ================= ADMITTANCE CONTROLLER STEP =================
            # The admittance step is performed here on the physics thread: make sure the frequency of this thread is 500Hz
            # if self.trigger_admittance == True:
            frequency = 500.0  # Hz: on isaac is possible thanks to this harware
            dt = 1 / frequency
            #tcp
            # T_tcp = XFormPrim("/World/Robot/wrist_3_link").get_world_pose()
            # # quat_tcp = R.from_quat(T_tcp[1])
            # robot_tcp_pose = (T_tcp[0],pr.matrix_from_quaternion(T_tcp[1]))
            #tcp
            T_tcp = XFormPrim("/World/Robot/wrist_3_link").get_world_pose()
            # quat_tcp = R.from_quat(T_tcp[1])
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
            # reading2 = self.robot._articulation_view._physics_view.get_force_sensor_forces()
            wrist_forces = reading[10]
            R_tcp_articulation_euler = R.from_euler('xyz',[-1.5707963, 0, -1.5707963])
            R_tcp_articulation = R_tcp_articulation_euler.as_matrix()
            # print(reading)
            # print(f"Force: {wrist_forces[:3]}, Torque: {wrist_forces[3:]}")
            # wrist_forces = np.array([wrist_forces[1]/1,-wrist_forces[2]/1,wrist_forces[0]/1,wrist_forces[4]/1,-wrist_forces[5]/1,wrist_forces[3]/1])
            self.wrist_forces = np.array([-wrist_forces[0],-wrist_forces[1],-wrist_forces[2],-wrist_forces[3],-wrist_forces[4],-wrist_forces[5]])
            # self.wrist_forces = copy.deepcopy(wrist_forces)
            # self.plot.publish(np.hstack((f,mu)).tolist())
            self.plot.publish(wrist_forces.tolist())

            # Forces are given in the articulation frame, rotate forces to TCP frame (necessary just in simulation) 
            f_w = self.wrist_forces[:3]
            mu_w = self.wrist_forces[3:]
            
            # use wrench transform to place the force torque in the tip.
            mu_tip, f_tip = wrench_trans(mu_w, f_w, self.T_tcp_tip)

            # rotate forces to base frame
            R_base_tip = T_base_tip[:3, :3]
            f_base_tip = R_base_tip @ f_tip

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
            base_tcp_out_ur_pose = get_transform_as_pose(T_base_tcp_out)

            # set position target of the robot
            print("output of the admittance controller: ", pt.pq_from_transform(T_base_tcp_out))
            world_tcp_out_pose = self.T_w_b@T_base_tcp_out
            # print("world_tcp_out_pose: ", world_tcp_out_pose )
            world_tcp_out_pose_pq = pt.pq_from_transform(world_tcp_out_pose)
            desired_pose = ut.get_pose3(world_tcp_out_pose_pq[:3],rot_quat=world_tcp_out_pose_pq[3:])
            
            # Move to desired pose
            self.robot.movePose3(desired_pose)

            self.counter = self.counter + dt
        return 
    
    def add_plate(self, prim_path, position, orientation):
        add_reference_to_stage(usd_path= ADRIAN_NUCLEUS_ASSETS + "cranfield_benchmark_bigger_holes.usd",prim_path=prim_path)
        XFormPrim(prim_path=prim_path,position=position,orientation=orientation)
        return
    
    
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