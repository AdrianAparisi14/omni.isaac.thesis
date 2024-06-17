import sys
import matplotlib.pyplot as plt
import os
import csv
import ast
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, 
    QMainWindow, 
    QVBoxLayout, 
    QHBoxLayout,
    QPushButton, 
    QLabel, 
    QWidget,
    QTabWidget,
    QTabBar,
    QListWidget,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QSizePolicy,
    QSplitter,
    )
from datetime import datetime
import argparse

success = True  # Example outcome (replace with your actual outcome)

class CustomTabBar(QTabBar):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setExpanding(False)
        self.setTabsClosable(False)
        

class MainWindow(QMainWindow):
    # def __init__(self):
    #     super().__init__()
        
    #     # Parse command-line arguments
    #     self.parse_arguments()
        
    #     # Initialize a list to store the filenames of the previously loaded files
    #     self.loaded_files = []

    #     self.setWindowTitle("Assembly Verification")
    #     # self.setFixedSize(QSize(1920, 1080))
    #     central_widget = QWidget()
    #     self.setCentralWidget(central_widget)
    #     layout = QVBoxLayout(central_widget)
        
    #     # Add custom tab bar
    #     custom_tab_bar = CustomTabBar()
    #     for item in ["recordings", "History", "Settings", "About"]:
    #         custom_tab_bar.addTab(item)

    #     layout.addWidget(custom_tab_bar)
        
    #     # Add list widget to show CSV files
    #     self.file_list_widget = QListWidget()
    #     layout.addWidget(self.file_list_widget)
    #     self.load_csv_files()
        
    #     # Add table widget to display CSV content
    #     self.table_widget = QTableWidget()
    #     layout.addWidget(self.table_widget)
    #     # Set stretch factor of table widget to 1
    #     self.table_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    #     layout.setStretchFactor(self.table_widget, 1)
        
    #     # Create a container widget for the plot
    #     self.plot_container = QWidget()
    #     layout.addWidget(self.plot_container)

    #     # Create a matplotlib Figure and a FigureCanvas
    #     self.plot_figure = Figure()
    #     self.plot_canvas = FigureCanvas(self.plot_figure)

    #     # Add the FigureCanvas to the plot container
    #     plot_layout = QVBoxLayout(self.plot_container)
    #     plot_layout.addWidget(self.plot_canvas)

    #     # Add outcome label
    #     self.outcome_label = QLabel("Outcome: ")
    #     layout.addWidget(self.outcome_label)

    #     # Add date and time label
    #     # current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     self.time_label = QLabel("Date and Time: ")
    #     layout.addWidget(self.time_label)
        
    #     # Create a QTimer
    #     self.timer = QTimer(self)
    #     # Connect the timeout signal of the QTimer to the function to update folder content
    #     self.timer.timeout.connect(self.update_folder_content)
    #     # Set the interval for the QTimer (e.g., refresh every 5 seconds)
    #     self.timer.start(5000)  # Refresh every 5000 milliseconds (5 seconds)
    
    def __init__(self):
        super().__init__()

        # Parse command-line arguments
        self.parse_arguments()

        # Initialize a list to store the filenames of the previously loaded files
        self.loaded_files = []

        self.setWindowTitle("Assembly Verification")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QVBoxLayout(central_widget)

        # Add custom tab bar
        custom_tab_bar = CustomTabBar()
        for item in ["recordings", "History", "Settings", "About"]:
            custom_tab_bar.addTab(item)

        main_layout.addWidget(custom_tab_bar)

        # Create a splitter widget
        splitter_main = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter_main)

        # Left side layout
        left_layout = QVBoxLayout()

        # Add list widget to show CSV files
        self.file_list_widget = QListWidget()
        left_layout.addWidget(self.file_list_widget)
        self.load_csv_files()

        # Add left layout to splitter
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        splitter_main.addWidget(left_widget)

        # Create a splitter widget for the right side
        splitter_right = QSplitter(Qt.Vertical)

        # Right side layout for table widget
        right_top_layout = QVBoxLayout()

        # Add table widget to display CSV content
        self.table_widget = QTableWidget()
        right_top_layout.addWidget(self.table_widget)
        self.table_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_top_layout.setStretchFactor(self.table_widget, 1)

        # Add right top layout to splitter
        right_top_widget = QWidget()
        right_top_widget.setLayout(right_top_layout)
        splitter_right.addWidget(right_top_widget)

        # Right side layout for plot widget
        right_bottom_layout = QVBoxLayout()

        # Create a container widget for the plot
        self.plot_container = QWidget()
        right_bottom_layout.addWidget(self.plot_container)

        # Create a matplotlib Figure and a FigureCanvas
        self.plot_figure = Figure()
        self.plot_canvas = FigureCanvas(self.plot_figure)

        # Add the FigureCanvas to the plot container
        plot_layout = QVBoxLayout(self.plot_container)
        plot_layout.addWidget(self.plot_canvas)

        # Add outcome label with fixed size
        self.outcome_label = QLabel("Outcome: ")
        self.outcome_label.setFixedHeight(40)  # Set fixed height
        right_bottom_layout.addWidget(self.outcome_label)

        # Add date and time label with fixed size
        self.time_label = QLabel("Date and Time: ")
        self.time_label.setFixedHeight(40)  # Set fixed height
        right_bottom_layout.addWidget(self.time_label)

        # Add right bottom layout to splitter
        right_bottom_widget = QWidget()
        right_bottom_widget.setLayout(right_bottom_layout)
        splitter_right.addWidget(right_bottom_widget)

        # Add splitter_right to splitter_main
        splitter_main.addWidget(splitter_right)

        # Set sizes of splitters
        splitter_main.setSizes([200, 600])
        splitter_right.setSizes([400, 200])

        # Create a QTimer
        self.timer = QTimer(self)
        # Connect the timeout signal of the QTimer to the function to update folder content
        self.timer.timeout.connect(self.update_folder_content)
        # Set the interval for the QTimer (e.g., refresh every 5 seconds)
        self.timer.start(3000)  # Refresh every 5000 milliseconds (5 seconds)
                
    def update_folder_content(self):
        if self.assembly_type == "pick_and_place":
            folder_path = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/script/csv_validation_pick_place"
        elif self.assembly_type == "snap":
            folder_path = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/script/csv_validation"
        
        # Filter out only CSV files
        current_csv_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.csv')]

        # Find new files that are not in the loaded_files list
        new_files = [filename for filename in current_csv_files if filename not in self.loaded_files]

        # Load the new files
        for filename in new_files:
            # Load the new file (example: add it to a list widget)
            self.file_list_widget.insertItem(0, filename)

        # Update the loaded_files list with the current files
        self.loaded_files = current_csv_files   
    
      
    def load_csv_files(self):
        # Folder containing CSV files
        if self.assembly_type == "pick_and_place":
            self.folder_path = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/script/csv_validation_pick_place"
        elif self.assembly_type == "snap":
            self.folder_path = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/script/csv_validation"
        
        # Read the CSV file into a list of lists
        # Check if the folder exists
        if os.path.isdir(self.folder_path):
            # Get a list of CSV files in the folder
            csv_files = sorted([file for file in os.listdir(self.folder_path) if file.endswith(".csv")], reverse=True)
            self.loaded_files = csv_files

            # Add CSV files to the list widget
            for file in csv_files:
                self.file_list_widget.addItem(file)
                
            # Connect item selection signal to slot
            self.file_list_widget.itemClicked.connect(self.display_file_contents)
            self.file_list_widget.itemClicked.connect(self.plot_force)
        

    def display_file_contents(self, item):
        # Get selected file name
        self.file_name = item.text()
        
        # Update the date label
        if self.file_name:
            # Get the date from the first file name
            date_str = self.file_name.split('data')[1].split('.csv')[0].strip()
            date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
            formatted_date = date_obj.strftime('%Y-%m-%d %H:%M:%S')
            self.time_label.setStyleSheet("font-size: 30pt;")
            self.time_label.setText("Date and Time: " + formatted_date)
        else:
            self.time_label.clear()

        # Read the files and create dataframes
        with open(os.path.join(self.folder_path, self.file_name), 'r') as file:
            lines = file.readlines()
            length_series = len(lines)
        # Check if the last line contains a valid label
        try:
            label = int(lines[-1].strip())
        except ValueError:
            print(f"Error: Invalid label in file {self.file_name}")

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
                print(f"Error: Unable to process line {line} in file {self.file_name}: {e}")

        # Extract forces and positions
        forces = [item[0][0] for item in data]
        positions = [item[1][0] for item in data]
        # Extract the label from the last line
        self.label = int(lines[-1].strip())  # Assuming the label is an integer
        # Create a DataFrame for each file
        file_df = pd.DataFrame({'Forces': forces, 'Positions': positions}) #, 'Label': label})

        # Create separate DataFrames for Force and Position
        force_df = pd.DataFrame(file_df['Forces'].tolist(), columns=['Force_1', 'Force_2', 'Force_3', 'Torque_1', 'Torque_2', 'Torque_3'])
        position_df = pd.DataFrame(file_df['Positions'].tolist(), columns=['Position_1', 'Position_2', 'Position_3', 'Orientation_1', 'Orientation_2', 'Orientation_3'])

        if self.label == 1:
            self.outcome_label.setStyleSheet("color: green; font-size: 40pt;")
            self.outcome_label.setText("SUCCESS")
        else:
            self.outcome_label.setStyleSheet("color: red; font-size: 40pt;")
            self.outcome_label.setText("FAILURE")

        # Concatenate Force_3
        self.Force_Z = np.array(force_df['Force_3'].tolist())

        # Display CSV content in a table widget
        self.display_csv_content(force_df, position_df, length_series)

    def display_csv_content(self, force, position, rows):
        num_rows = rows
        num_columns = force.shape[1] + position.shape[1] # 6 for forces and 6 for positions

        self.table_widget.clear()

        self.table_widget.setRowCount(num_rows)
        self.table_widget.setColumnCount(num_columns)

        # Set column headers
        headers = ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "X", "Y", "Z", "Rx", "Ry", "Rz"]
        self.table_widget.setHorizontalHeaderLabels(headers * (num_columns // len(headers)))

        for (index1, row1), (index2, row2) in zip(force.iterrows(), position.iterrows()):
            # Populate table with force values
            for col_index1, col_name1 in enumerate(force.columns):
                value1 = force.iloc[index1, col_index1]
                item1 = QTableWidgetItem(str(value1))
                self.table_widget.setItem(index1, col_index1, item1)

            # Populate table with position values
            for col_index2, col_name2 in enumerate(position.columns):
                value2 = position.iloc[index1, col_index2]
                item2 = QTableWidgetItem(str(value2))
                self.table_widget.setItem(index1, col_index2 + 6, item2)
        
        # Adjust column widths to fit contents
        for col in range(num_columns):
            self.table_widget.resizeColumnToContents(col)  

    def plot_force(self):
        # Clear the plot container before plotting new data
        self.plot_figure.clear()  

        # Create a new plot and add it to the plot container
        ax = self.plot_figure.add_subplot(111)
        ax.plot(self.Force_Z, marker='o', markersize=1)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Force in Z')
        ax.set_title('Force Recording')
        ax.grid(True)


        self.plot_canvas.draw()
        
    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='GUI for verification of assembly processes. Indicate which type of assembly has to be verified.')
        parser.add_argument('--assembly_type', dest='assembly_type', default='pick_and_place',
                            help='Choose type of assembly: (pick_and_place/snap)')

        args = parser.parse_args()

        self.assembly_type = args.assembly_type

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()