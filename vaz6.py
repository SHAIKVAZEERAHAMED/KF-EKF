import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
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

# Create Kalman filter
num_states = 6  # Number of states to track (velocity_x, velocity_y, latitude, longitude)
dim_z = 6  # Number of measurements (velocity_x, velocity_y, latitude, longitude)

kf = KalmanFilter(dim_x=num_states, dim_z=dim_z)

# Initialize Kalman filter
initial_state = np.array([
    data[velocity_columns[0]].iloc[0],
    data[velocity_columns[1]].iloc[0],
    data[position_columns[0]].iloc[0],
    data[position_columns[1]].iloc[0],
    0,
    0
])
initial_covariance = np.eye(num_states)

kf.x = initial_state
kf.P = initial_covariance

# Define the state transition matrix
dt = 1.0  # Time step
kf.F = np.array([
    [1, 0, dt, 0, 0.5*dt**2, 0],
    [0, 1, 0, dt, 0, 0.5*dt**2],
    [0, 0, 1, 0, dt, 0],
    [0, 0, 0, 1, 0, dt],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
])

# Define the measurement matrix
kf.H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
])

# Define the measurement noise
measurement_noise = np.eye(dim_z) * 0.1  # Adjust the measurement noise covariance as needed
kf.R = measurement_noise

# Track the object using the Kalman filter
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
    kf.predict()

    # Update the object's state based on the measurement
    kf.update(measurement)

    # Save the estimated state
    track.append(kf.x.copy())

track = np.array(track)

# Plot the trajectory
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.plot(original_trajectory_position[:, 1], original_trajectory_position[:, 0], label='Original Position Trajectory')
plt.plot(track[:, 3], track[:, 2], label='Estimated Position Trajectory')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Object Position Trajectory')
plt.legend()

plt.subplot(122)
plt.plot(original_trajectory_velocity[:, 0], label='Velocity X (Original)')
plt.plot(original_trajectory_velocity[:, 1], label='Velocity Y (Original)')
plt.plot(track[:, 0], label='Velocity X (Estimated)')
plt.plot(track[:, 1], label='Velocity Y (Estimated)')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Object Velocity')
plt.legend()

plt.tight_layout()
plt.show()
