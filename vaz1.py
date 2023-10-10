import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt

# File path of the dataset
file_path = r"C:\Users\SHAIK VAZEER AHAMED\Downloads\AUV- Snapir\Tedelyne DVL\teledyne_navigator_measurements.xlsx"

# Load dataset
data = pd.read_excel(file_path)

# Convert non-numeric values to NaN
data[['latitude', 'longitute', 'Errror [m/sec]', 'w speed [m/sec]', 'v speed [m/sec]', 'u speed [m/sec]']] = data[['latitude', 'longitute', 'Errror [m/sec]', 'w speed [m/sec]', 'v speed [m/sec]', 'u speed [m/sec]']].apply(pd.to_numeric, errors='coerce')

# Convert 'Time' column to total seconds
data['Time'] = data['Time'].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second)

# Create Kalman filter
kf = KalmanFilter(dim_x=6, dim_z=3)

# Initialize Kalman filter
initial_state = np.array([0, 0, 0, 0, 0, 0])
initial_covariance = np.eye(6)

kf.x = initial_state
kf.P = initial_covariance

# Define measurement function and matrix
kf.H = np.eye(3, 6)

# Define transition matrix and process noise
dt = 1.0
kf.F = np.array([[1, 0, 0, dt, 0, 0],
                 [0, 1, 0, 0, dt, 0],
                 [0, 0, 1, 0, 0, dt],
                 [0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1]])

kf.Q = np.eye(6) * 0.01

# Apply Kalman filter to the dataset
filtered_data = []
for index, row in data.iterrows():
    measurement = np.array([row['u speed [m/sec]'], row['v speed [m/sec]'], row['w speed [m/sec]']]).astype(np.float64)
    kf.predict()
    kf.update(measurement)
    filtered_data.append(kf.x[:3])

# Plot the filtered data
filtered_data = np.array(filtered_data)

# Plot the original and filtered data
plt.figure(figsize=(12, 6))

# Plot original data
plt.subplot(1, 2, 1)
plt.plot(data['Time'], data['u speed [m/sec]'], label='Original u speed')
plt.plot(data['Time'], data['v speed [m/sec]'], label='Original v speed')
plt.plot(data['Time'], data['w speed [m/sec]'], label='Original w speed')
plt.xlabel('Time')
plt.ylabel('Velocity [m/sec]')
plt.title('Original Data')
plt.legend()

# Plot filtered data
plt.subplot(1, 2, 2)
plt.plot(data['Time'], filtered_data[:, 0], label='Filtered u speed')
plt.plot(data['Time'], filtered_data[:, 1], label='Filtered v speed')
plt.plot(data['Time'], filtered_data[:, 2], label='Filtered w speed')
plt.xlabel('Time')
plt.ylabel('Velocity [m/sec]')
plt.title('Filtered Data')
plt.legend()

plt.tight_layout()
plt.show()
