# ===Set initial target===
        
        # circle
        # self.counter = 0.0
        # might better use another position
        # x_desired = self.get_circle_target(self.T_base_tip_pos_init, self.counter)
        # print("x desired: ", x_desired)
        # T_base_tip_circle = pt.transform_from_pq(np.hstack((x_desired, self.T_base_tip_quat_init)))
        # T_base_tcp_circle = T_base_tip_circle @ self.T_tip_tcp
        # print("T_world_tcp_circle: ", T_base_tcp_circle)

        # # Use add_jtraj to move to the initial point on the circle.
        # T_base_tcp_circle = self.T_w_b@T_base_tip
        # T_w_tpc_circle = self.T_w_b@T_base_tcp_circle
        # print("T_w_tpc_circle", T_w_tpc_circle)
        # T_w_tpc_circle_pq = pt.pq_from_transform(T_base_tcp_circle)
        # # add_jtraj work on the world coordinate system, this will be sent to perform the trajectory      
        
        # # Move to the startin pose of the movement
        # desired_pose = ut.get_pose3(T_w_tpc_circle_pq[:3],rot_quat=T_w_tpc_circle_pq[3:])
        # self.robot.movePose3(desired_pose)
        
        
        
# def get_circle_target(self, pose, timestep, radius=0.075, freq=0.5):
    #     import copy
    #     circ_target = copy.deepcopy(pose)
    #     circ_target[0] = pose[0] + radius * np.cos((2 * np.pi * freq * timestep))
    #     circ_target[1] = pose[1] + radius * np.sin((2 * np.pi * freq * timestep))
    #     return circ_target
    


def physics_step(self):
        from omni.sdu.utils.utilities.math_util import wrench_trans, get_robot_pose_as_transform, get_pose_as_transform, get_transform_as_pq, \
                                                                get_transform_as_ur_pose, R_to_rotvec, get_transform_as_pose
        from scipy.spatial.transform import Rotation as R
        import quaternion
        from pytransform3d import transformations as pt
        
        # ================= ADMITTANCE CONTROLLER STEP =================
        # The admittance step is performed here on the physics thread: make sure the frequency of this thread is 500Hz
        # if self.trigger_admittance == True:
        frequency = 500.0  # Hz
        dt = 1 / frequency
        T_tcp = XFormPrim("/World/Robots/robotA/wrist_3_link").get_world_pose()
        quat = R.from_quat(T_tcp[1])
        robot_tcp_pose = (T_tcp[0],quat.as_matrix())
        T_w_tcp = get_pose_as_transform(robot_tcp_pose)
        
        # robot_base_pose = XFormPrim("/World/Robots/robotA").get_world_pose()
        # quat = R.from_quat(robot_base_pose[1])
        # robot_base_pose = (robot_base_pose[0],quat.as_matrix())
        # T_w_b = get_pose_as_transform(robot_base_pose)
        # T_w_b_inv = np.linalg.inv(T_w_b)
        T_b_tcp = self.T_w_b_inv@T_w_tcp
        
        T_base_tcp = T_b_tcp
        T_base_tip = T_base_tcp @ self.T_tcp_tip

        # get tip in base as position and quaternion
        T_base_tip_pq = pt.pq_from_transform(T_base_tip)
        T_base_tip_pos = T_base_tip_pq[0:3]
        T_base_tip_quat = T_base_tip_pq[3:7]

        # get current robot force-torque in tcp frame (measured at tcp)
        reading = self.robot.get_force_measurement()
        wrist_forces = reading[9]
        # print(reading)
        # print("Force in the wrist: ", wrist_forces)
        print(f"Force: {wrist_forces[:3]}, Torque: {wrist_forces[3:]}")
        wrist_forces = np.array([wrist_forces[1],wrist_forces[2],wrist_forces[0],wrist_forces[4],wrist_forces[5],wrist_forces[3]])
        # self.plot.publish(wrist_forces.tolist())
        # self.plot.publish(wrist_forces)

        # rotate forces from worlf frame to TCP frame 
        f_w = wrist_forces[:3]
        mu_w = wrist_forces[3:]
        
        f_tcp = T_w_tcp[:3,:3]@f_w
        mu_tcp = T_w_tcp[:3,:3]@mu_w
        
        
        # use wrench transform to place the force torque in the tip.
        # mu_tip, f_tip = wrench_trans(mu_w, f_w, T_base_tip)
        mu_tip, f_tip = wrench_trans(mu_tcp, f_tcp, self.T_tcp_tip)

        # rotate forces back to base frame
        R_base_tip = T_base_tip[:3, :3]
        f_base_tip = R_base_tip @ f_tip

        # the input position and orientation is given as tip in base
        self.adm_controller.pos_input = T_base_tip_pos
        self.adm_controller.rot_input = quaternion.from_float_array(T_base_tip_quat)
        self.adm_controller.q_input = self.robot.get_joint_positions()
        self.adm_controller.ft_input = np.hstack((f_base_tip, mu_tip))

        # get circle target
        x_desired = self.get_circle_target(self.T_base_tip_pos_init, self.counter)
        self.adm_controller.set_desired_frame(x_desired, quaternion.from_float_array(self.T_base_tip_quat_init))

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
        print("output of the admittance controller: ", base_tcp_out_ur_pose)
        world_tcp_out_pose = self.T_w_b@T_base_tcp_out
        world_tcp_out_pose_pq = pt.pq_from_transform(world_tcp_out_pose)
        desired_pose = ut.get_pose3(world_tcp_out_pose_pq[:3],rot_quat=world_tcp_out_pose_pq[3:])
        
        # task.add_jtraj(desired_pose = ut.get_pose3(world_tcp_out_pose_pq[:3],rot_quat=world_tcp_out_pose_pq[3:]), time_step=self.physics_dt)
        self.robot.movePose3(desired_pose)
        
        self.counter = self.counter + dt
        return