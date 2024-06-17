from omni.isaac.manipulators.grippers import ParallelGripper
import numpy as np
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.prims import get_prim_at_path
import omni.kit.commands
from pxr import Usd, Sdf
from omni.sdu.utils.utilities import utils as ut
from omni.isaac.core.utils.nucleus import get_assets_root_path 
import pandas as pd
from datetime import datetime


from scipy.signal import butter, lfilter
from scipy.stats import multivariate_normal

from scipy.interpolate import interp1d

from omni.sdu.ai.utilities import utils as ut_ai


NOVO_NUCLEUS_ASSETS = "omniverse://sdur-nucleus.tek.sdu.dk/Projects/novo/Assets/"

# Functions for simulation purposes:

def attach_fingertip_snap(gripper="robotiq"):
    """Attaches the gripper to the robot.
        This function assumes the robot is a UR robot defined by Isaac Sim standards.

    Args:

    Returns:
    """    
    gripper_ = gripper
    if gripper_ == "robotiq":
        omni.kit.commands.execute('RemoveRelationshipTarget',
            relationship=get_prim_at_path("/World/Robot/tool0/hand_e/tcp/FixedJoint").GetRelationship('physics:body1'),
            target=Sdf.Path("/World/Robot/tool0/hand_e/tcp"))
        omni.kit.commands.execute('AddRelationshipTarget',
            relationship=get_prim_at_path("/World/Robot/tool0/hand_e/tcp/FixedJoint").GetRelationship('physics:body1'),
            target=Sdf.Path("/World/Robot/tool0/hand_e/add_tcp/fingertip/hook_body"))
    elif gripper_== "crg200":
        omni.kit.commands.execute('RemoveRelationshipTarget',
            relationship=get_prim_at_path("/World/Robot/tool0/crg200/tcp/FixedJoint").GetRelationship('physics:body1'),
            target=Sdf.Path("/World/Robot/tool0/crg200/tcp"))
        omni.kit.commands.execute('AddRelationshipTarget',
            relationship=get_prim_at_path("/World/Robot/tool0/crg200/tcp/FixedJoint").GetRelationship('physics:body1'),
            target=Sdf.Path("/World/Robot/tool0/crg200/add_tcp/fingertip/hook_body"))


def random_point_from_2d_gaussian(mean, cov_matrix):
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

def random_orientation_quaternion(mean_quaternion, std_dev_angle):
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

def interpolate_z_force(time_series_1, time_series_2, csv_file_interpolated):
    
        # ===== Approach for only obtain the FZ interpolated =====
        # Read force values from both CSV files
        force_values_ts1, length_series1 = ut_ai.get_dataframe_from_directory(time_series_1)
        force_values_ts2, length_series2 = ut_ai.get_dataframe_from_directory(time_series_2)
        
        # Create separate DataFrames for Force and Position
        force_df = pd.DataFrame(force_values_ts1['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
        force_values_ts1_z = np.array(force_df['Force_3'].tolist())

        # Determine the sampling interval for time series 1
        num_samples_ts1 = length_series1 - 1
        num_samples_ts2 = length_series2 - 1
        sampling_interval_ts1 = (num_samples_ts2 - 1) / (num_samples_ts1 - 1)

        # Generate the timestamps for time series 1
        timestamps_ts1 = np.arange(0, num_samples_ts1) * sampling_interval_ts1

        # Interpolate the force values from time series 1 to match the timestamps of time series 2
        interpolated_force_values_ts1 = interp1d(timestamps_ts1, force_values_ts1_z, kind='linear')(np.arange(0, num_samples_ts2))

        # Generate the timestamps for time series 2
        timestamps_ts2 = np.arange(0, num_samples_ts2)  # Assuming each recording serves as a timestamp
        
        # Create a DataFrame with timestamps and interpolated force values
        df_interpolated = pd.DataFrame({'force': interpolated_force_values_ts1}) # Interpolated_force_values_ts1 contains the force values from time series 1 interpolated to match the timestamps of time series 2

        # Save the DataFrame to a new CSV file
        df_interpolated.to_csv('../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_simulation/csv_snap_admittance_interpolated/interpolated_data.csv', index=False)


def interpolate_timeseries(time_series_1, time_series_2, csv_file_interpolated):
    """
    Generate an interpolated timeseries to match time_series_1 and time_series_2 in time steps length

    Parameters:
    - time_series_1: time series to be interpolated (short time series)
    - time_series_2: time series used as a reference to be compared in length

    Returns:
    - df_interpolated: dataframe of interpolated force values
    """
    
    # ===== Approach for obtain all dataset interpolated =====
    # Interpolate the entire dataframe (forces and poses)
    df_ts1, length_series1 = ut_ai.get_dataframe_from_directory(time_series_1)
    df_ts2, length_series2 = ut_ai.get_dataframe_from_directory(time_series_2)
    
    # Interpolate positions
    df_positions = pd.DataFrame(df_ts1['Positions'].tolist(), columns=['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6'])
    interpolated_positions = pd.DataFrame()
    for column in df_positions.columns:
        # Access column values using item_df[column]
        column_values = df_positions[column]
        num_samples_ts1 = length_series1 - 1
        num_samples_ts2 = length_series2 - 1
        
        # Determine the sampling interval for time series 1
        sampling_interval_ts1 = (num_samples_ts2 - 1) / (num_samples_ts1 - 1)
        
        # Generate the timestamps for time series 1
        timestamps_ts1 = np.arange(0, num_samples_ts1) * sampling_interval_ts1
        
        # Interpolate the force values from time series 1 to match the timestamps of time series 2
        interpolated_values = interp1d(timestamps_ts1, column_values, kind='linear')(np.arange(0, num_samples_ts2))
        interpolated_positions[column] = interpolated_values

    # Interpolate forces
    df_forces = pd.DataFrame(df_ts1['Forces'].tolist(), columns=['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6'])
    interpolated_forces = pd.DataFrame()
    for column in df_forces.columns:
        # Access column values using item_df[column]
        column_values = df_forces[column]
        num_samples_ts1 = length_series1 - 1
        num_samples_ts2 = length_series2 - 1
        
        # Determine the sampling interval for time series 1
        sampling_interval_ts1 = (num_samples_ts2 - 1) / (num_samples_ts1 - 1)
        
        # Generate the timestamps for time series 1
        timestamps_ts1 = np.arange(0, num_samples_ts1) * sampling_interval_ts1
        
        # Interpolate the force values from time series 1 to match the timestamps of time series 2
        interpolated_values = interp1d(timestamps_ts1, column_values, kind='linear')(np.arange(0, num_samples_ts2))
        interpolated_forces[column] = interpolated_values

    # Combine interpolated positions and forces
    combined_data = pd.concat([interpolated_positions, interpolated_forces], axis=1)

    # Write to CSV with desired format
    # Path to the csv file to store force-torque data interpolated
    current_time = datetime.now()
    csv_file = csv_file_interpolated
    # csv_file = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_simulation/csv_snap_position_interpolated/data" + str(current_time) + ".csv"
    # csv_file = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/statistical_analysis/csv_snap_sim_position/data" + str(current_time) + ".csv"
    with open(csv_file, 'w') as f:
        for index, row in combined_data.iterrows():
            position_values = row[:len(df_positions.columns)].values
            force_values = row[len(df_positions.columns):].values
            # f.write(f"{force_values}\t{position_values}\n")
            pair = [force_values.tolist(), position_values.tolist()]
            force = [pair[0]]
            pos = [pair[1]]
            f.write(f'{force}\t{pos}\n')

    return # df_interpolated



# Not used functions
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