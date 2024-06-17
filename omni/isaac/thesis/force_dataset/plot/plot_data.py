# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import os
import ast
import pandas as pd

def plot_last_assembly(data_directory):
    file_path = data_directory
    # Read the CSV file into a list of lists
    with open(file_path, 'r') as file:
        lines = file.readlines()
        length_series = len(lines)
    # Convert each line to a list using ast.literal_eval
    data = []
    for line in lines:  # Exclude the last line
        try:
            # Split the line into forces and positions
            forces_str, positions_str = line.strip().split('\t')
            forces = ast.literal_eval(forces_str)
            positions = ast.literal_eval(positions_str)
            data.append((forces, positions))
        except (SyntaxError, ValueError, TypeError) as e:
            print(f"Error: Unable to process line {line} in file {file_path}: {e}")

    # Extract forces and positions
    forces = [item[0][0] for item in data]
    positions = [item[1][0] for item in data]
    # Extract the label from the last line
    # Create a DataFrame for each file
    full_df = pd.DataFrame({'Forces': forces, 'Positions': positions})
    # return full_df, length_series

    # Create separate DataFrames for Force and Position
    force_df = pd.DataFrame(full_df['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
    # position_df = pd.DataFrame(full_df['Position'].tolist(), columns=['Position_1', 'Position_2', 'Position_3', 'Orientation_1', 'Orientation_2', 'Orientation_3'])

    # Assuming force_df is your feature matrix and labels_df is your target vector
    # Concatenate Force_3 column with other relevant features if necessary
    X = np.array(force_df['Force_3'].tolist())
    X = X.reshape((len(X) // (length_series), (length_series)))

    # Extract the third component (index 2) from each sublist
    force_component = X[-1,:]
    print(force_component)

    # Plot the data
    plt.plot(force_component, marker='o', markersize=1)
    plt.xlabel('Data Point')
    plt.ylabel('Force (Third Component)')
    plt.title('Force Data')
    plt.grid(True)
    plt.show()



def plot_full():
    data_directory = "../../../../../Documents/sdu.extensions.2023.1.0.hotfix/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_real_robot_novo/csv_real_robot_admittance_novo"

    # List to store dataframes from each CSV file
    dataframes = []

    # Loop through CSV files in the directory
    for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_directory, filename)
            # Read the CSV file into a list of lists
            with open(file_path, 'r') as file:
                lines = file.readlines()
                length_series = len(lines)
            # Check if the last line contains a valid label
            try:
                label = int(lines[-1].strip())
            except ValueError:
                print(f"Error: Invalid label in file {filename}")
                continue

            # Convert each line to a list using ast.literal_eval
            data = []
            for line in lines[:-1]:  # Exclude the last line
                try:
                    # Split the line into forces and positions
                    forces_str, positions_str = line.strip().split('\t')
                    forces = ast.literal_eval(forces_str)
                    positions = ast.literal_eval(positions_str)
                    data.append((forces, positions))
                except (SyntaxError, ValueError, TypeError) as e:
                    print(f"Error: Unable to process line {line} in file {filename}: {e}")

            # Extract forces and positions
            forces = [item[0][0] for item in data]
            positions = [item[1][0] for item in data]
            # Extract the label from the last line
            label = int(lines[-1].strip())  # Assuming the label is an integer
            # Create a DataFrame for each file
            file_df = pd.DataFrame({'Forces': forces, 'Positions': positions, 'Label': label})
            # Append the DataFrame to the list
            dataframes.append(file_df)


    # Concatenate dataframes into one
    full_df = pd.concat(dataframes, ignore_index=True)

    # Create separate DataFrames for Force and Position
    force_df = pd.DataFrame(full_df['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
    # position_df = pd.DataFrame(full_df['Position'].tolist(), columns=['Position_1', 'Position_2', 'Position_3', 'Orientation_1', 'Orientation_2', 'Orientation_3'])

    y = []  # Initialize the target variable

    for index, row in full_df.iterrows():
        if (index + 1) % (length_series - 1) == 0:
            y.append(row['Label'])

    # Assuming force_df is your feature matrix and labels_df is your target vector
    # Concatenate Force_3 column with other relevant features if necessary
    X = np.array(force_df['Force_3'].tolist())
    X = X.reshape((len(X) // (length_series - 1), (length_series - 1)))

    # Extract the third component (index 2) from each sublist
    force_component = X[-1,:]
    print(force_component)

    # Plot the data
    plt.plot(force_component, marker='o', markersize=1)
    plt.xlabel('Data Point')
    plt.ylabel('Force (Third Component)')
    plt.title('Force Data')
    plt.grid(True)
    plt.show()
    
def plot_all_scv():
    data_directory = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_pick_place/sim/admittance_interpolated"

    # Loop through CSV files in the directory
    for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_directory, filename)
            print(filename)
            # Read the CSV file into a list of lists
            with open(file_path, 'r') as file:
                lines = file.readlines()
                length_series = len(lines)
            # Check if the last line contains a valid label
            try:
                label = int(lines[-1].strip())
            except ValueError:
                print(f"Error: Invalid label in file {file_path}")

            # Convert each line to a list using ast.literal_eval
            data = []
            for line in lines[:-1]:  # Exclude the last line
                try:
                    # Split the line into forces and positions
                    forces_str, positions_str = line.strip().split('\t')
                    forces = ast.literal_eval(forces_str)
                    positions = ast.literal_eval(positions_str)
                    data.append((forces, positions))
                except (SyntaxError, ValueError, TypeError) as e:
                    print(f"Error: Unable to process line {line} in file {file_path}: {e}")

            # Extract forces and positions
            forces = [item[0][0] for item in data]
            positions = [item[1][0] for item in data]
            # Extract the label from the last line
            label = int(lines[-1].strip())  # Assuming the label is an integer
            # Create a DataFrame for each file
            file_df = pd.DataFrame({'Forces': forces, 'Positions': positions, 'Label': label})

            # Create separate DataFrames for Force and Position
            force_df = pd.DataFrame(file_df['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
            # position_df = pd.DataFrame(full_df['Position'].tolist(), columns=['Position_1', 'Position_2', 'Position_3', 'Orientation_1', 'Orientation_2', 'Orientation_3'])

            y = []  # Initialize the target variable

            for index, row in file_df.iterrows():
                if (index + 1) % (length_series - 1) == 0:
                    y.append(row['Label'])

            # Assuming force_df is your feature matrix and labels_df is your target vector
            # Concatenate Force_3 column with other relevant features if necessary
            X = np.array(force_df['Force_3'].tolist())

            # Plot the data
            plt.plot(X, marker='o', markersize=1)
            plt.xlabel('Data Point')
            plt.ylabel('Force (Third Component)')
            plt.title('Force Data')
            plt.grid(True)
            plt.show()


def plot_one_csv():
    # data_directory = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_real_robot_sdu/csv_real_robot_position_all_correct/data2024-04-25 08:31:42.286459.csv"
    # data_directory = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/csv_pick_place/real/admittance_all_correct/data2024-05-01 10:06:37.155940.csv"
    # data_directory = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/plot/comparison_sim_real_pick_place/data2024-06-30 09:38:00.711842.csv"
    data_directory = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.sdu.ur/omni/sdu/ur/scripts/force_verification/train_data_clutch/data2024-06-10 09:48:38.479110.csv"

    file_path = data_directory
    # Read the CSV file into a list of lists
    with open(file_path, 'r') as file:
        lines = file.readlines()
        length_series = len(lines)
    # Check if the last line contains a valid label
    try:
        label = int(lines[-1].strip())
    except ValueError:
        print(f"Error: Invalid label in file {file_path}")

    # Convert each line to a list using ast.literal_eval
    data = []
    for line in lines[:-1]:  # Exclude the last line
        try:
            # Split the line into forces and positions
            forces_str, positions_str = line.strip().split('\t')
            forces = ast.literal_eval(forces_str)
            positions = ast.literal_eval(positions_str)
            data.append((forces, positions))
        except (SyntaxError, ValueError, TypeError) as e:
            print(f"Error: Unable to process line {line} in file {file_path}: {e}")

    # Extract forces and positions
    forces = [item[0][0] for item in data]
    positions = [item[1][0] for item in data]
    # Extract the label from the last line
    label = int(lines[-1].strip())  # Assuming the label is an integer
    # Create a DataFrame for each file
    file_df = pd.DataFrame({'Forces': forces, 'Positions': positions, 'Label': label})

    # Create separate DataFrames for Force and Position
    force_df = pd.DataFrame(file_df['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
    # position_df = pd.DataFrame(full_df['Position'].tolist(), columns=['Position_1', 'Position_2', 'Position_3', 'Orientation_1', 'Orientation_2', 'Orientation_3'])

    y = []  # Initialize the target variable

    for index, row in file_df.iterrows():
        if (index + 1) % (length_series - 1) == 0:
            y.append(row['Label'])

    # Assuming force_df is your feature matrix and labels_df is your target vector
    # Concatenate Force_3 column with other relevant features if necessary
    X = np.array(force_df['Force_3'].tolist())
    
    plot_directory = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/plot/cases_verification_sdu'
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    # Plot the data
    plt.figure(figsize=(11, 4))
    plt.plot(X, marker='o', markersize=1)
    plt.xlabel('Data Point')
    plt.ylabel('Force in Z axis (N)')
    plt.title('Force Data')
    
    # # Add vertical lines
    # vertical_lines = [2500, 2995, 3125, 3600]
    # for x in vertical_lines:
    #     plt.axvline(x=x, color='red', linestyle='--')

    # # Add numbers between the vertical lines
    # annotations = [(2500 + 2995) / 2, (2995 + 3125) / 2, (3125 + 3600) / 2]
    # for i, x in enumerate(annotations):
    #     plt.text(x, max(X) * 0.9, str(i + 1), color='black', fontsize=13, ha='center', va='center', weight='bold')

    
    plt.grid(True)
    
    # plt.savefig(os.path.join(plot_directory, 'real_asymtot_snap.png'), format='png', dpi=300, bbox_inches='tight')
    # # Save the plot as a PDF file
    # plt.savefig(os.path.join(plot_directory, 'real_asymtot_snap.pdf'), format='pdf', bbox_inches='tight')
    # # Save the plot as an EPS file
    # plt.savefig(os.path.join(plot_directory, 'real_asymtot_snap.eps'), format='eps', bbox_inches='tight')
    
    plt.show()

            
def plot_overlapped_escenarios():
    data_directory = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/plot/comparison_sim_real_pick_place"
    # data_directory = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/plot/cases_verification_novo_real_device"
    # data_directory = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/plot/cases_verification_pick_place"
    # data_directory = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/plot/comparison_sim_real"
    # data_directory = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/plot/cases_verification_sdu"
    
    # List to store dataframes from each CSV file
    dataframes = []
    scenarios = []

    # Loop through CSV files in the directory
    for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_directory, filename)
            # Read the CSV file into a list of lists
            with open(file_path, 'r') as file:
                lines = file.readlines()
                length_series = len(lines)
            # Check if the last line contains a valid label
            try:
                label = int(lines[-1].strip())
            except ValueError:
                print(f"Error: Invalid label in file {filename}")
                continue

            # Convert each line to a list using ast.literal_eval
            data = []
            for line in lines[:-1]:  # Exclude the last line
                try:
                    # Split the line into forces and positions
                    forces_str, positions_str = line.strip().split('\t')
                    forces = ast.literal_eval(forces_str)
                    positions = ast.literal_eval(positions_str)
                    data.append((forces, positions))
                except (SyntaxError, ValueError, TypeError) as e:
                    print(f"Error: Unable to process line {line} in file {filename}: {e}")

            # Extract forces and positions
            forces = [item[0][0] for item in data]
            positions = [item[1][0] for item in data]
            # Extract the label from the last line
            label = int(lines[-1].strip())  # Assuming the label is an integer
            # Create a DataFrame for each file
            file_df = pd.DataFrame({'Forces': forces, 'Positions': positions, 'Label': label})
            # Append the DataFrame to the list
            dataframes.append(file_df)


    # Concatenate dataframes into one
    full_df = pd.concat(dataframes, ignore_index=True)

    # Create separate DataFrames for Force and Position
    force_df = pd.DataFrame(full_df['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
    # position_df = pd.DataFrame(full_df['Position'].tolist(), columns=['Position_1', 'Position_2', 'Position_3', 'Orientation_1', 'Orientation_2', 'Orientation_3'])

    y = []  # Initialize the target variable

    for index, row in full_df.iterrows():
        if (index + 1) % (length_series - 1) == 0:
            y.append(row['Label'])

    # Assuming force_df is your feature matrix and labels_df is your target vector
    # Concatenate Force_3 column with other relevant features if necessary
    X = np.array(force_df['Force_3'].tolist())
    scenarios = X.reshape((len(X) // (length_series - 1), (length_series - 1)))
    print(scenarios.shape)
    
    # Plot the data
    plt.figure(figsize=(7, 5))
    for i, force_component in enumerate(scenarios.tolist()):
        # plt.plot(force_component, label=f"Scenario {i+1}", marker='o', markersize=1)
        if i == 0:
            plt.plot(force_component, label="Sim")
        elif i == 1:
            plt.plot(force_component, label="Real")
        elif i == 2:
            plt.plot(force_component, label="Exceeded insertion force")
        
        # if i == 0:
        #     plt.plot(force_component, label="Exceeded insertion target/clogged piece at the bottom", marker='o', markersize=1)
        # elif i == 1:
        #     plt.plot(force_component, label="Housing missing", marker='o', markersize=1)
        # elif i == 2:
        #     plt.plot(force_component, label="Snap-fit part missing", marker='o', markersize=1,color='grey')
        # elif i == 3:
        #     plt.plot(force_component, label="Correct assembly", marker='o', markersize=1, color='green')
        # elif i == 4:
        #     plt.plot(force_component, label="Part not fully inserted", marker='o', markersize=1, color='red')
        
        # if i == 0:
        #     plt.plot(force_component, label="Correct assembly", marker='o', markersize=1, color='green')
        # elif i == 1:
        #     plt.plot(force_component, label="Part missing", marker='o', markersize=1, color='grey')
        # elif i == 2:
        #     plt.plot(force_component, label="exceeded insertion target/\nclogged piece at the bottom", marker='o', markersize=1,color='orange')


    plot_directory = '../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/plot/comparison_sim_real_pick_place'
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # Plot the data
    # plt.plot(X, marker='o', markersize=1)
    plt.xlabel('Data Point')
    plt.ylabel('Force in Z axis (N)')
    plt.title('Force Data')
    plt.legend()
    plt.grid(True)
    # plt.savefig(os.path.join(data_directory, 'overlapped_scenarios.png'))
    # Save the plot as a high-resolution PNG file
    plt.savefig(os.path.join(data_directory, 'sim_real_pp.png'), format='png', dpi=300, bbox_inches='tight')
    # Save the plot as a PDF file
    plt.savefig(os.path.join(data_directory, 'sim_real_pp.pdf'), format='pdf', bbox_inches='tight')
    # Save the plot as an EPS file
    plt.savefig(os.path.join(data_directory, 'sim_real_pp.eps'), format='eps', bbox_inches='tight')
    plt.show()
    
    
plot_one_csv()