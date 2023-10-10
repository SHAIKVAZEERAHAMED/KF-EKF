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
num_states = 8  # Number of states to track (Acc_X, Acc_Y, Acc_Z, Long, Lat, Gyro_X, Gyro_Y, Gyro_Z)
dim_z = 8  # Number of measurements (Acc_X, Acc_Y, Acc_Z, Long, Lat, Gyro_X, Gyro_Y, Gyro_Z)

kf = KalmanFilter(dim_x=num_states, dim_z=dim_z)

# Initialize Kalman filter
initial_state = data.iloc[0][columns].values.astype(np.float64)
initial_covariance = np.eye(num_states) * 0.1

kf.x = initial_state
kf.P = initial_covariance

# Define state transition matrix and process noise
dt = 1.0
kf.F = np.eye(num_states)
kf.Q = np.eye(num_states) * 0.01

# Define measurement matrix
kf.H = np.eye(dim_z, num_states)

# Define measurement noise covariance
kf.R = np.eye(dim_z) * 0.1  # Adjust the value to control the filtering of measurement noise

# Extract original data for plotting
original_acc = data[['Acc_X', 'Acc_Y', 'Acc_Z']].values
original_gyro = data[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].values
original_lat = data['Lat'].values
original_long = data['Long'].values

# Track the object using the Kalman filter
filtered_states = []
for _, row in data.iterrows():
    measurement = row[columns].values.astype(np.float64)
    kf.predict()
    kf.update(measurement)
    filtered_states.append(kf.x.copy())

# Convert filtered states to numpy array
filtered_states = np.array(filtered_states)

# Extract filtered data for plotting
filtered_acc = filtered_states[:, :3]
filtered_gyro = filtered_states[:, 5:8]
filtered_lat = filtered_states[:, 4]
filtered_long = filtered_states[:, 3]

# Plot the original and filtered trajectories
plt.figure(figsize=(18, 12))

# Plot original acceleration
plt.subplot(2, 4, 1)
plt.plot(original_acc[:, 0], label='Original Acc_X')
plt.plot(original_acc[:, 1], label='Original Acc_Y')
plt.plot(original_acc[:, 2], label='Original Acc_Z')
plt.xlabel('Index')
plt.ylabel('Acceleration')
plt.title('Original Acceleration')
plt.legend()

# Plot original gyroscope
plt.subplot(2, 4, 2)
plt.plot(original_gyro[:, 0], label='Original Gyro_X')
plt.plot(original_gyro[:, 1], label='Original Gyro_Y')
plt.plot(original_gyro[:, 2], label='Original Gyro_Z')
plt.xlabel('Index')
plt.ylabel('Angular Velocity')
plt.title('Original Gyro')
plt.legend()

# Plot original latitude
plt.subplot(2, 4, 3)
plt.plot(original_lat, label='Original Latitude')
plt.xlabel('Index')
plt.ylabel('Latitude')
plt.title('Original Latitude')
plt.legend()

# Plot original longitude
plt.subplot(2, 4, 4)
plt.plot(original_long, label='Original Longitude')
plt.xlabel('Index')
plt.ylabel('Longitude')
plt.title('Original Longitude')
plt.legend()

# Plot filtered acceleration
plt.subplot(2, 4, 5)
plt.plot(filtered_acc[:, 0], label='Filtered Acc_X')
plt.plot(filtered_acc[:, 1], label='Filtered Acc_Y')
plt.plot(filtered_acc[:, 2], label='Filtered Acc_Z')
plt.xlabel('Index')
plt.ylabel('Acceleration')
plt.title('Filtered Acceleration')
plt.legend()

# Plot filtered gyroscope
plt.subplot(2, 4, 6)
plt.plot(filtered_gyro[:, 0], label='Filtered Gyro_X')
plt.plot(filtered_gyro[:, 1], label='Filtered Gyro_Y')
plt.plot(filtered_gyro[:, 2], label='Filtered Gyro_Z')
plt.xlabel('Index')
plt.ylabel('Angular Velocity')
plt.title('Filtered Gyro')
plt.legend()

# Plot filtered latitude
plt.subplot(2, 4, 7)
plt.plot(filtered_lat, label='Filtered Latitude')
plt.xlabel('Index')
plt.ylabel('Latitude')
plt.title('Filtered Latitude')
plt.legend()

# Plot filtered longitude
plt.subplot(2, 4, 8)
plt.plot(filtered_long, label='Filtered Longitude')
plt.xlabel('Index')
plt.ylabel('Longitude')
plt.title('Filtered Longitude')
plt.legend()

plt.tight_layout()
plt.show()
