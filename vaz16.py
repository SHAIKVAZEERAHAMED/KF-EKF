import pandas as pd
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
import matplotlib.pyplot as plt

# File path of the dataset
file_path = r"C:\Users\SHAIK VAZEER AHAMED\Downloads\AUV- Snapir\Tedelyne DVL\teledyne_navigator_measurements.xlsx"

# Load dataset
data = pd.read_excel(file_path)

# Extract relevant columns for tracking
velocity_columns = ['u speed [m/sec]', 'v speed [m/sec]']

# Convert non-numeric values to NaN for velocity columns
data[velocity_columns] = data[velocity_columns].apply(pd.to_numeric, errors='coerce')

# Filter out rows with insufficient data for tracking
data = data.dropna(subset=velocity_columns)

# Original trajectory
original_trajectory = data[velocity_columns].values

# Create Extended Kalman Filter
num_states = 4  # Number of states to track (velocity_x, velocity_y, position_x, position_y)
dim_z = 2  # Number of measurements (velocity_x, velocity_y)

ekf = ExtendedKalmanFilter(dim_x=num_states, dim_z=dim_z)

# Define the system dynamics function (transition matrix) for the EKF
def system_dynamics(x, dt):
    # The system dynamics can be modeled as a constant velocity motion model
    # Assuming the state contains velocity in x and y directions, and position in x and y directions,
    # the dynamics can be defined as:
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    return np.dot(F, x)

# Define the measurement function and its Jacobian matrix for the EKF
def measurement_function(x):
    # The measurement function can be a direct measurement of velocity in x and y directions
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    return np.dot(H, x).flatten()

# Define the Jacobian matrix of the measurement function
def measurement_jacobian(x):
    # The Jacobian matrix of the measurement function can be a 2x4 matrix
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    return H

# Set the system dynamics and measurement function for the EKF
ekf.f = system_dynamics
ekf.h = measurement_function

# Initialize EKF
initial_state = np.array([
    data[velocity_columns[0]].iloc[0],
    data[velocity_columns[1]].iloc[0],
    data['latitude'].iloc[0],
    data['longitute'].iloc[0]
])
initial_covariance = np.eye(num_states)

ekf.x = initial_state
ekf.P = initial_covariance

# Define the process noise covariance matrix
process_noise = np.eye(num_states) * 0.1  # Adjust the process noise covariance as needed
ekf.Q = process_noise

# Define the measurement noise covariance matrix
measurement_noise = np.eye(dim_z) * 0.1  # Adjust the measurement noise covariance as needed
ekf.R = measurement_noise

# Track the object using the Extended Kalman Filter
track = []
for _, row in data.iterrows():
    measurement = np.array([
        row[velocity_columns[0]],
        row[velocity_columns[1]]
    ])

    # Predict the object's state
    ekf.predict()

    # Update the object's state based on the measurement
    ekf.update(measurement, HJacobian=measurement_jacobian, Hx=measurement_function)

    # Save the estimated state
    track.append(ekf.x.copy())

track = np.array(track)

# Plot the original trajectory
plt.plot(original_trajectory[:, 0], original_trajectory[:, 1], label='Original Trajectory')

# Plot the estimated trajectory
plt.plot(track[:, 0], track[:, 1], label='Estimated Trajectory')

plt.xlabel('Velocity X')
plt.ylabel('Velocity Y')
plt.title('Object Trajectory before and after Extended Kalman Filtering')
plt.legend()
plt.show()
