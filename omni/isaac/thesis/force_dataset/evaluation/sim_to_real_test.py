from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True}) # we can also run as headless.

from scipy import stats
import itertools
from omni.sdu.ai.utilities import utils as ut_ai
import pandas as pd
import numpy as np


# Get the sets from their folders: only correct assemblies are being compared
# First: get the full datasets from the folder
path_data_real = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/statistical_analysis/csv_snap_real_position"
path_data_sim = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/statistical_analysis/csv_snap_sim_position"

df_sim, length_series_sim = ut_ai.get_fulldatafrane_from_directory(path_data_sim)
df_real, length_series_real = ut_ai.get_fulldatafrane_from_directory(path_data_real)

# Transform the dataframes into set of lists
force_df_sim = pd.DataFrame(df_sim['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
force_df_real = pd.DataFrame(df_real['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])

# Process only Z-force
array_sim = np.array(force_df_sim['Force_3'].tolist())
array_real = np.array(force_df_real['Force_3'].tolist())

# Reshape X to have dimensions (number of assemblies, ((length_series)), 1)
array_real = array_real.reshape((len(array_real) // (length_series_real-1),(length_series_real-1)))
array_sim = array_sim.reshape((len(array_sim) // (length_series_sim-1), (length_series_sim-1)))

set1 = array_sim.tolist()  # List of distributions in set 1
set2 = array_real.tolist()  # List of distributions in set 2

# print(set1[-1])

# Number of comparisons
# num_comparisons = len(set1) * len(set2)

# # Perform pairwise comparison and Bonferroni correction
# for dist1, dist2 in itertools.product(set1, set2):
#     statistic, p_value = stats.ks_2samp(dist1, dist2)
#     # Bonferroni correction
#     p_value_adjusted = p_value * num_comparisons
#     print("KS Statistic:", statistic)
#     print("P-value (adjusted):", p_value_adjusted)
#     if p_value_adjusted < 0.05:
#         print("Reject Null Hypothesis")
#     else:
#         print("Fail to Reject Null Hypothesis")
        




statistic, p_value = stats.ks_2samp(set1[-1], set2[-1])

# Print results
print("KS Statistic:", statistic)
print("P-value:", p_value)