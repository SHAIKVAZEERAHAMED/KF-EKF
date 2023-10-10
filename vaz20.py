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
position_columns = ['latitude', 'longitute']

# Convert non-numeric values to NaN for velocity and position columns
data[velocity_columns + position_columns] = data[velocity_columns + position_columns].apply(pd.to_numeric, errors='coerce')

# Filter out rows with insufficient data for tracking
data = data.dropna(subset=velocity_columns + position_columns)

# Original trajectory
original_trajectory_position = data[position_columns].values
original_trajectory_velocity = data[velocity_columns].values

# Create Extended Kalman Filter
num_states = 6  # Number of states to track (velocity_x, velocity_y, latitude, longitude)
dim_z = 6  # Number of measurements (velocity_x, velocity_y, latitude, longitude)

ekf = ExtendedKalmanFilter(dim_x=num_states, dim_z=dim_z)

# Initialize Extended Kalman Filter
initial_state = np.array([
    data[velocity_columns[0]].iloc[0],
    data[velocity_columns[1]].iloc[0],
    data[position_columns[0]].iloc[0],
    data[position_columns[1]].iloc[0],
    0,
    0
])
initial_covariance = np.eye(num_states)

ekf.x = initial_state
ekf.P = initial_covariance

# Define the state transition function
def f(x):
    dt = 1.0  # Time step
    F = np.array([
        [1, 0, dt, 0, 0.5 * dt ** 2, 0],
        [0, 1, 0, dt, 0, 0.5 * dt ** 2],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    return F.dot(x)

ekf.f = f

# Define the measurement function
def h(x):
    return x

ekf.h = h

# Define the Jacobian of the measurement function
def HJacobian(x):
    return np.eye(dim_z)

ekf.HJacobian = HJacobian

# Define the measurement function
def Hx(x):
    return x

ekf.Hx = Hx

# Define the measurement noise
measurement_noise = np.eye(dim_z) * 0.1  # Adjust the measurement noise covariance as needed
ekf.R = measurement_noise

# Track the object using the Extended Kalman Filter
track = []
for _, row in data.iterrows():
    measurement = np.array([
        row[velocity_columns[0]],
        row[velocity_columns[1]],
        row[position_columns[0]],
        row[position_columns[1]],
        0,
        0
    ])

    # Predict the object's state
    ekf.predict()

    # Update the object's state based on the measurement
    ekf.update(measurement, HJacobian, Hx)

    # Save the estimated state
    track.append(ekf.x.copy())

track = np.array(track)

# Plot the trajectory
plt.figure()

plt.subplot(121)
plt.plot(original_trajectory_position[:, 1], original_trajectory_position[:, 0], label='Original Position Trajectory')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Object Position Trajectory')
plt.legend()

plt.subplot(122)
plt.plot(original_trajectory_velocity[:, 0], label='Velocity X (Original)')
plt.plot(original_trajectory_velocity[:, 1], label='Velocity Y (Original)')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Object Velocity')
plt.legend()

plt.figure()
plt.subplot(121)
plt.plot(track[:, 3], track[:, 2], label='Estimated Position Trajectory')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Estimated Object Position Trajectory')
plt.legend()

plt.subplot(122)
plt.plot(track[:, 0], label='Velocity X (Estimated)')
plt.plot(track[:, 1], label='Velocity Y (Estimated)')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Estimated Object Velocity')
plt.legend()

plt.show()