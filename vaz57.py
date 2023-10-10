import pandas as pd
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
import matplotlib.pyplot as plt
# KF Magnetic data 0
import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt

# File path of the dataset
file_path1 = r"C:\Users\SHAIK VAZEER AHAMED\Downloads\magnetic_data.csv"

# Load the data from CSV file
data1 = pd.read_csv(file_path1)

# Create Kalman filter
num_states1 = 4  # Number of states to track (Pos(Mx), Pos(My), Pos(Mz), dis_m, ang_m)
dim_z1 = 4  # Number of measurements (Pos(Mx), Pos(My), Pos(Mz), dis_m, ang_m)

kf1 = KalmanFilter(dim_x=num_states1, dim_z=dim_z1)

# Initialize Kalman filter
initial_state1 = np.array([data1['Pos(Mx)'].iloc[0], data1['Pos(My)'].iloc[0], data1['dis_m'].iloc[0], data1['ang_m'].iloc[0]])
initial_covariance1 = np.eye(num_states1) * 0.1  # Adjust the initial covariance as needed

kf1.x = initial_state1
kf1.P = initial_covariance1

# Define the state transition matrix
dt = 1  # Time step
kf1.F = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

# Define the measurement matrix
kf1.H = np.eye(dim_z1)

# Define the process noise covariance matrix
process_noise1 = np.eye(num_states1) * 0.01  # Adjust the process noise covariance as needed
kf1.Q = process_noise1

# Define the measurement noise covariance matrix
measurement_noise1 = np.eye(dim_z1) * 0.1  # Adjust the measurement noise covariance as needed
kf1.R = measurement_noise1

# Apply Kalman filter to estimate the true values
estimated_values1 = []
for _, row in data1.iterrows():
    measurement1 = np.array([
    row['Pos(Mx)'],
    row['Pos(My)'],
    row['dis_m'],
    row['ang_m']
    ])

    # Predict the object's state
    kf1.predict()

    # Update the object's state based on the measurement
    kf1.update(measurement1)

    # Save the estimated state
    estimated_values1.append(kf1.x.copy())


estimated_values1 = np.array(estimated_values1)


# File path of the dataset
file_path = r"C:\Users\SHAIK VAZEER AHAMED\Downloads\magnetic_data.csv"

# Load the data from CSV file
data = pd.read_csv(file_path)

# Create Extended Kalman Filter
num_states = 4  # Number of states to track (Pos(Mx), Pos(My), Pos(Mz), dis_m, ang_m)
dim_z = 4  # Number of measurements (Pos(Mx), Pos(My), Pos(Mz), dis_m, ang_m)

ekf = ExtendedKalmanFilter(dim_x=num_states, dim_z=dim_z)

# Initialize Extended Kalman Filter
initial_state = np.array([data['Pos(Mx)'].iloc[0], data['Pos(My)'].iloc[0], data['dis_m'].iloc[0], data['ang_m'].iloc[0]])
initial_covariance = np.eye(num_states) * 0.1  # Adjust the initial covariance as needed

ekf.x = initial_state
ekf.P = initial_covariance

# Define the state transition function
def state_transition_function(x, dt):
    # Nonlinear state transition function
    x_pred = np.zeros_like(x)
    x_pred[0] = x[0] + dt * x[1]  # Nonlinear propagation for Pos(Mx)
    x_pred[1] = x[1]  # No change for Pos(My)
    x_pred[2] = x[2]  # No change for dis_m
    x_pred[3] = x[3]  # No change for ang_m
    return x_pred

# Define the measurement function
def measurement_function(x):
    return x

def measurement_jacobian(x):
    jacobian = np.eye(dim_z)
    return jacobian

# Set the state transition, measurement, and measurement jacobian functions
ekf.f = state_transition_function
ekf.h = measurement_function
ekf.H = measurement_jacobian


# Define the process noise covariance matrix
process_noise = np.eye(num_states) * 0.01  # Adjust the process noise covariance as needed
ekf.Q = process_noise

# Define the measurement noise covariance matrix
measurement_noise = np.eye(dim_z) * 0.1  # Adjust the measurement noise covariance as needed
ekf.R = measurement_noise

# Apply Extended Kalman Filter to estimate the true values
estimated_values = []
for _, row in data.iterrows():
    measurement = np.array([
        row['Pos(Mx)'],
        row['Pos(My)'],
        row['dis_m'],
        row['ang_m']
    ])

    # Predict the object's state
    ekf.predict()

    # Update the object's state based on the measurement
    ekf.update(measurement, ekf.H, ekf.h)

    # Save the estimated state
    estimated_values.append(ekf.x.copy())

estimated_values = np.array(estimated_values)

# Plot the original and estimated values
plt.figure()
plt.plot(data1['Pos(Mx)'], data1['Pos(My)'], 'g*-', label='Original')
plt.plot(estimated_values1[:, 0], estimated_values1[:, 1], 'yo-', label='Estimated kf')

plt.xlabel('Position (Mx)')
plt.ylabel('Position (My)')
plt.title('Original and Estimated Positions')
plt.legend()

plt.plot(data['Pos(Mx)'], data['Pos(My)'], 'g*-', label='Original')
plt.plot(estimated_values[:, 0], estimated_values[:, 1], 'y*-', label='Estimated ekf')

plt.xlabel('Position (Mx)')
plt.ylabel('Position (My)')
plt.title('Original and Estimated Positions')
plt.legend()

plt.figure()
plt.plot(data1['sno_m'], data1['dis_m'], 'yo-', label='Original dis_m')
plt.plot(data1['sno_m'], estimated_values1[:, 2], 'ro-', label='Estimated dis_m kf')

plt.xlabel('Sample Number')
plt.ylabel('dis_m ')
plt.title('Original and Estimated Distance')
plt.legend()

plt.plot(data['sno_m'], data['dis_m'], 'yo-', label='Original dis_m')
plt.plot(data['sno_m'], estimated_values[:, 2], 'r*-', label='Estimated dis_m ekf')

plt.xlabel('Sample Number')
plt.ylabel('dis_m ')
plt.title('Original and Estimated Distance')
plt.legend()

plt.figure()
plt.plot(data1['sno_m'], data1['ang_m'], 'mo-', label='Original ang_m')
plt.plot(data1['sno_m'], estimated_values1[:, 3], 'ro-', label='Estimated ang_m kf')
plt.xlabel('Sample Number')
plt.ylabel('KF')
plt.legend()

plt.plot(data['sno_m'], data['ang_m'], 'mo-', label='Original ang_m')
plt.plot(data['sno_m'], estimated_values[:, 3], 'r*-', label='Estimated ang_m ekf')
plt.xlabel('Sample Number')
plt.ylabel('ang_m')
plt.title('Original and Estimated Angle')
plt.legend()
plt.show()
