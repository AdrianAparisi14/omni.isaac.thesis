#launch Isaac Sim before any other imports
#default first two lines in any standalone application
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True}) # we can also run as headless.

import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import statistics


# load parameters
# Read the JSON files contining the Forces and Rotations
file_path_1 = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/script/tool_calibration/sim/Fcalib1.json"
file_path_2 = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/script/tool_calibration/sim/Fcalib2.json"
file_path_3 = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/script/tool_calibration/sim/Fcalib3.json"

file_paths = [file_path_1, file_path_2, file_path_3]

# Initialize lists to store Forces and Moments for each file
Forces = []
Rotations = []

# Read data from the JSON files
for file in file_paths:
    with open(file, "r") as json_file:
        data = json.load(json_file)
        Forces.append(data["Forces"])
        Rotations.append(data["Rotations"])

# Initialize lists to store Forces and Moments after conversion to NumPy arrays
F_list = []
R_list = []

# Convert Forces and Moments to NumPy arrays
for file_forces, file_rotations in zip(Forces, Rotations):
    F_array = np.array([np.array(force) for force in file_forces])
    R_array = np.array([np.array(moment) for moment in file_rotations])
    F_list.append(F_array)
    R_list.append(R_array)
    
# -0.5366040789163522, -0.2637460773675756, 0.8015572775314], [-0.6516355135628076, -0.47399167005535825, -0.5922018694515142]]]
R1 = np.array(R_list[0][-1]).T
R1[:, [0, 1]] = R1[:, [1, 0]]
R1[:, 0] *= -1
R2 = np.array(R_list[1][-1]).T
R2[:, [0, 1]] = R2[:, [1, 0]]
R2[:, 0] *= -1
R3 = np.array(R_list[2][-1]).T
R3[:, [0, 1]] = R3[:, [1, 0]]
R3[:, 0] *= -1

print(R1, R2, R3)
# print(F_list[0])

# Separate Forces into components for the first file
F_component1 = []
M_component1 = []
for F in F_list[0]:
    F_component1.append(np.array(F[:3]))
    M_component1.append(np.array(F[3:]))

# Separate Forces into components for the second file
F_component2 = []
M_component2 = []
for F in F_list[1]:
    F_component2.append(np.array(F[:3]))
    M_component2.append(np.array(F[3:]))

# Separate Forces into components for the third file
F_component3 = []
M_component3 = []
for F in F_list[2]:
    F_component3.append(np.array(F[:3]))
    M_component3.append(np.array(F[3:]))
    

# # Calculate means for the first file
# F1 = np.mean(np.array(F_component1)[399:500], axis=0)
# M1 = np.mean(np.array(M_component1)[399:500], axis=0)

# # Calculate means for the second file
# F2 = np.mean(np.array(F_component2)[399:500], axis=0)
# M2 = np.mean(np.array(M_component2)[399:500], axis=0)

# # Calculate means for the third file
# F3 = np.mean(np.array(F_component3)[399:500], axis=0)
# M3 = np.mean(np.array(M_component3)[399:500], axis=0)

# Calculate means for the first file
F1 = np.mean(np.array(F_component1), axis=0)
M1 = np.mean(np.array(M_component1), axis=0)

# Calculate means for the second file
F2 = np.mean(np.array(F_component2), axis=0)
M2 = np.mean(np.array(M_component2), axis=0)

# Calculate means for the third file
F3 = np.mean(np.array(F_component3), axis=0)
M3 = np.mean(np.array(M_component3), axis=0)

# Example usage:
print("Mean Force from File 1:", F1)
print("Mean Moment from File 1:", M1)
print("Mean Force from File 2:", F2)
print("Mean Moment from File 2:", M2)
print("Mean Force from File 3:", F3)
print("Mean Moment from File 3:", M3)


# F1 = mean(F1(400:500,:))';
# F2 = mean(F2(400:500,:))';
# F3 = mean(F3(400:500,:))';
# M1 = mean(M1(400:500,:))';
# M2 = mean(M2(400:500,:))';
# M3 = mean(M3(400:500,:))';
# % F1 = mean(squeeze(F1(:,:,400:500))')';
# % F2 = mean(squeeze(F2(:,:,900:1000))')';
# % F3 = mean(squeeze(F3(:,:,1400:1500))')';
# %
# % M1 = mean(squeeze(M(:,:,400:500))')';
# % M2 = mean(squeeze(M(:,:,900:1000))')';
# % M3 = mean(squeeze(M(:,:,1400:1500))')';
# % rotation matrices do not need to be filtered
# % transposed R matrices!!!
# %(transformacija iz world k.s. (gravitacija) v k.s. robota)
# % transformation from world c.f. (gravity) to robot c.f.



# % matrix of 3 measurements R
# RI=[R1,eye(3); R2,eye(3); R3,eye(3)];
# Constructing the identity matrix
eye_matrix = np.eye(3)

# Concatenating R1, R2, and R3 horizontally with the identity matrix
RI = np.concatenate((np.concatenate((R1, eye_matrix), axis=1),
                     np.concatenate((R2, eye_matrix), axis=1),
                     np.concatenate((R3, eye_matrix), axis=1)), axis=0)

print("RI: ", RI)


# % matrix of 3 F measurements
# Fmeas=[F1, F2, F3]
Fmeas = np.hstack((F1, F2, F3)).T
# Fmeas = np.concatenate((F1.reshape(-1, 1), F2.reshape(-1, 1), F3.reshape(-1, 1)), axis=0)
print("Fmeas: ", Fmeas)


# %izracun Fg, Foff
# force_compensation=inv(RI'*RI)*RI'*Fmeas;
# Fg=force_compensation(1:3)
# Foff=force_compensation(4:6)
# Calculate force compensation
force_compensation = np.linalg.inv(RI.T @ RI) @ RI.T @ Fmeas

# Extract Fg and Foff
Fg = force_compensation[:3]
Foff = force_compensation[3:]
# Fg = np.array((-0.0409, 0.0451, -15.5696))
# Foff = np.array((0.0035, 0.0020, 0.0127))

print("Fg:", Fg)
print("Foff:", Foff)



# % vector operator
# Fcomp = R1*Fg;
# Fvect_op1 = [0 -Fcomp(3) Fcomp(2);  Fcomp(3) 0 -Fcomp(1);  -Fcomp(2) Fcomp(1) 0];
# Fcomp = R2*Fg;
# Fvect_op2 = [0 -Fcomp(3) Fcomp(2);  Fcomp(3) 0 -Fcomp(1);  -Fcomp(2) Fcomp(1) 0];
# Fcomp = R3*Fg;
# Fvect_op3 = [0 -Fcomp(3) Fcomp(2);  Fcomp(3) 0 -Fcomp(1);  -Fcomp(2) Fcomp(1) 0];

# Calculate Fcomp for R1
Fcomp = np.dot(R1, Fg)
Fvect_op1 = np.array([[0, -Fcomp[2], Fcomp[1]],
                      [Fcomp[2], 0, -Fcomp[0]],
                      [-Fcomp[1], Fcomp[0], 0]])

# Calculate Fcomp for R2
Fcomp = np.dot(R2, Fg)
Fvect_op2 = np.array([[0, -Fcomp[2], Fcomp[1]],
                      [Fcomp[2], 0, -Fcomp[0]],
                      [-Fcomp[1], Fcomp[0], 0]])

# Calculate Fcomp for R3
Fcomp = np.dot(R3, Fg)
Fvect_op3 = np.array([[0, -Fcomp[2], Fcomp[1]],
                      [Fcomp[2], 0, -Fcomp[0]],
                      [-Fcomp[1], Fcomp[0], 0]])



# % matrika 3 meritev Fcomp v obliki vectorskega prod. in I
# % matrix of 3 measurements Fcomp in the shape of cross product in I
# MI=[-Fvect_op1,eye(3); -Fvect_op2,eye(3); -Fvect_op3,eye(3)];
# Concatenating arrays horizontally with the identity matrix
MI = np.concatenate((np.concatenate((-Fvect_op1, eye_matrix), axis=1),
                     np.concatenate((-Fvect_op2, eye_matrix), axis=1),
                     np.concatenate((-Fvect_op3, eye_matrix), axis=1)), axis=0)

print("MI: ", MI)


# % matrika 3 meritev F
# % matrix of 3 measurements of F
# Mmeas=[M1; M2; M3];
Mmeas = np.hstack((M1, M2, M3)).T
print("Mmeas: ", Mmeas)



# % izracun r in Moff
# % calculation of r in Moff
# torque_compensation=inv(MI'*MI)*MI'*Mmeas;
# r=torque_compensation(1:3)
# Moff=torque_compensation(4:6)
# Calculate torque compensation
torque_compensation = np.linalg.inv(MI.T @ MI) @ MI.T @ Mmeas

# Extract r and Moff
r = torque_compensation[:3]
Moff = torque_compensation[3:]

print("r:", r)
print("Moff:", Moff)

# disp('Comp. parameters saved')
# % Fg=Fg';
# % r = r';
# % Foff=Foff';
# % Moff=Moff';
# %save (['FcompIROS_Demo'], 'Fg', 'r', 'Foff', 'Moff')
# TSPA=0.02;
# %q0=[0;65;0;-65;0;-108;0]*pi/180;

# Combine data to store in a dictionary
data = {"Fg": Fg.tolist(), "r": r.tolist(), "Foff": Foff.tolist(), "Moff": Moff.tolist()}

# File path to store JSON data
data_directory = "../../../../../Documents/sdu.extensions.2023.1.1/exts/omni.isaac.thesis/omni/isaac/thesis/force_dataset/script/tool_calibration/sim"
file_path = data_directory + "/FComp_params.json"

# Writing data to the JSON file
with open(file_path, "w") as json_file:
    json.dump(data, json_file)

print("Data has been stored in " + data_directory)