import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt

# File path of the dataset
file_path = r"C:\Users\SHAIK VAZEER AHAMED\Downloads\Land vehicle\Stationary_MRU_2021.xlsx"

# Load dataset
data = pd.read_excel(file_path)

# Convert non-numeric values to NaN
numeric_columns = ['Magn_Z', 'Magn_Y', 'Magn_X', 'Acc_Z', 'Acc_Y', 'Acc_X', 'Gyro_Z', 'Gyro_Y', 'Gyro_X', 'Heading', 'Pitch', 'Roll', 'P_Bar', 'Altitude', 'Long', 'Lat']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Extract relevant columns for tracking
columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Long', 'Lat', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

# Filter out rows with insufficient data for tracking
data = data.dropna(subset=columns)

# Create Kalman filter
num_states = 8  # Number of states to track (acceleration_x, acceleration_y, acceleration_z, longitude, latitude, gyro_x, gyro_y, gyro_z)
dim_z = 8  # Number of measurements

kf = KalmanFilter(dim_x=num_states, dim_z=dim_z)

# Initialize Kalman filter
initial_state = np.zeros(num_states)
initial_covariance = np.eye(num_states)

kf.x = initial_state
kf.P = initial_covariance

# Define state transition matrix and process noise
dt = 1.0
kf.F = np.eye(num_states)
kf.Q = np.eye(num_states) * 0.01

# Define measurement matrix
kf.H = np.eye(dim_z, num_states)

# Track the object using the Kalman filter
filtered_states = []
for _, row in data.iterrows():
    measurement = row[columns].values.astype(np.float64)
    kf.predict()
    kf.update(measurement)
    filtered_states.append(kf.x.copy())

# Convert filtered states to numpy array
filtered_states = np.array(filtered_states)

# Plot the original and filtered trajectories
plt.figure(figsize=(18, 6))

# Plot original acceleration
plt.subplot(1, 3, 1)
plt.plot(data.index, data['Acc_X'], label='Original Acc_X')
plt.plot(data.index, data['Acc_Y'], label='Original Acc_Y')
plt.plot(data.index, data['Acc_Z'], label='Original Acc_Z')
plt.xlabel('Index')
plt.ylabel('Acceleration')
plt.title('Original Acceleration')
plt.legend()

# Plot original gyro
plt.subplot(1, 3, 2)
plt.plot(data.index, data['Gyro_X'], label='Original Gyro_X')
plt.plot(data.index, data['Gyro_Y'], label='Original Gyro_Y')
plt.plot(data.index, data['Gyro_Z'], label='Original Gyro_Z')
plt.xlabel('Index')
plt.ylabel('Angular Velocity')
plt.title('Original Gyro')
plt.legend()

# Plot original trajectory
plt.subplot(1, 3, 3)
plt.plot(data['Long'], data['Lat'], label='Original Trajectory')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Original Trajectory')
plt.legend()

plt.tight_layout()
plt.show()

# Plot the filtered trajectories
plt.figure(figsize=(18, 6))

# Plot filtered acceleration
plt.subplot(1, 3, 1)
plt.plot(data.index, filtered_states[:, 0], label='Filtered Acc_X')
plt.plot(data.index, filtered_states[:, 1], label='Filtered Acc_Y')
plt.plot(data.index, filtered_states[:, 2], label='Filtered Acc_Z')
plt.xlabel('Index')
plt.ylabel('Acceleration')
plt.title('Filtered Acceleration')
plt.legend()

# Plot filtered gyro
plt.subplot(1, 3, 2)
plt.plot(data.index, filtered_states[:, 5], label='Filtered Gyro_X')
plt.plot(data.index, filtered_states[:, 6], label='Filtered Gyro_Y')
plt.plot(data.index, filtered_states[:, 7], label='Filtered Gyro_Z')
plt.xlabel('Index')
plt.ylabel('Angular Velocity')
plt.title('Filtered Gyro')
plt.legend()

# Plot filtered trajectory
plt.subplot(1, 3, 3)
plt.plot(filtered_states[:, 3], filtered_states[:, 4], label='Filtered Trajectory')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Filtered Trajectory')
plt.legend()

plt.tight_layout()
plt.show()
