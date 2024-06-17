# Author: Adrian Aparisi Segui
# Thesis: Integrating AI-based approach for FailureDetection and Optimal Force ControlParameter Estimation in UR RobotAssembly Processes.
# Mail contact: adapa20@student.sdu.dk
#
# This extension is created in order to produce a virtual dataset
# with forces and torques of the UR5 TCP where the gap
# sim-to-real will be tested in order to apply AI
# techiques to, first, elaborate a time series analysis of the 
# assembly process; second, identify failures during the assembly;
# and third, find proper force control parameters to improve the 
# assembly process
#

# NOTE: Import here your extension examples to be propagated to ISAAC SIM Extensions startup
from omni.isaac.thesis.force_dataset.force_dataset import SceneForceDataset
from omni.isaac.thesis.force_dataset.force_dataset_extension import HelloWorldExtension