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

# Create Kalman filter
num_states = 9  # Number of states to track
dim_z = 3  # Number of measurements

kf = KalmanFilter(dim_x=num_states, dim_z=dim_z)

# Initialize Kalman filter
initial_state = np.zeros(num_states)
initial_covariance = np.eye(num_states)

kf.x = initial_state
kf.P = initial_covariance

# Define measurement function and matrix
kf.H = np.eye(dim_z, num_states)

# Define transition matrix and process noise
dt = 1.0
kf.F = np.eye(num_states)
kf.Q = np.eye(num_states) * 0.01

# Apply Kalman filter to the dataset
filtered_data = []
for index, row in data.iterrows():
    measurement = np.array([row['Acc_X'], row['Acc_Y'], row['Acc_Z']]).astype(np.float64)
    kf.predict()
    kf.update(measurement)
    filtered_data.append(kf.x[:dim_z])

# Convert filtered data to numpy array
filtered_data = np.array(filtered_data)

# Plot the original and filtered data
plt.figure(figsize=(12, 6))

# Plot original data
plt.subplot(1, 2, 1)
plt.plot(data.index, data['Acc_X'], label='Original Acc_X')
plt.plot(data.index, data['Acc_Y'], label='Original Acc_Y')
plt.plot(data.index, data['Acc_Z'], label='Original Acc_Z')
plt.xlabel('Index')
plt.ylabel('Acceleration')
plt.title('Original Data')
plt.legend()

# Plot filtered data
plt.subplot(1, 2, 2)
plt.plot(data.index, filtered_data[:, 0], label='Filtered Acc_X')
plt.plot(data.index, filtered_data[:, 1], label='Filtered Acc_Y')
plt.plot(data.index, filtered_data[:, 2], label='Filtered Acc_Z')
plt.xlabel('Index')
plt.ylabel('Acceleration')
plt.title('Filtered Data')
plt.legend()

plt.tight_layout()
plt.show()
