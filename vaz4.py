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

# Convert non-numeric values to NaN for velocity columns
data[velocity_columns] = data[velocity_columns].apply(pd.to_numeric, errors='coerce')

# Filter out rows with insufficient data for tracking
data = data.dropna(subset=velocity_columns)

# Original trajectory
original_trajectory = data[velocity_columns].values

# Create Kalman filter
num_states = 2  # Number of states to track (velocity_x, velocity_y)
dim_z = 2  # Number of measurements (velocity_x, velocity_y)

kf = KalmanFilter(dim_x=num_states, dim_z=dim_z)

# Initialize Kalman filter
initial_state = np.array([
    data[velocity_columns[0]].iloc[0],
    data[velocity_columns[1]].iloc[0]
])
initial_covariance = np.eye(num_states)

kf.x = initial_state
kf.P = initial_covariance

# Define the measurement matrix
kf.H = np.eye(dim_z)

# Define the measurement noise
measurement_noise = np.eye(dim_z) * 0.1  # Adjust the measurement noise covariance as needed
kf.R = measurement_noise

# Track the object using the Kalman filter
track = []
for _, row in data.iterrows():
    measurement = np.array([
        row[velocity_columns[0]],
        row[velocity_columns[1]]
    ])

    # Predict the object's state
    kf.predict()

    # Update the object's state based on the measurement
    kf.update(measurement)

    # Save the estimated state
    track.append(kf.x.copy())

track = np.array(track)

# Plot the original trajectory
plt.plot(original_trajectory[:, 0], original_trajectory[:, 1], label='Original Trajectory')

# Plot the estimated trajectory
plt.plot(track[:, 0], track[:, 1], label='Estimated Trajectory')

plt.xlabel('Velocity X')
plt.ylabel('Velocity Y')
plt.title('Object Trajectory before and after Kalman Filtering')
plt.legend()
plt.show()
