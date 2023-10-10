import pandas as pd
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
import matplotlib.pyplot as plt

# File path of the dataset
file_path = r"C:\Users\SHAIK VAZEER AHAMED\Downloads\Land vehicle\Stationary_MRU_2021.xlsx"

# Load dataset
data = pd.read_excel(file_path)

# Replace NaN values with previous valid values
data = data.fillna(method='ffill')

# Convert non-numeric values to NaN
numeric_columns = ['Magn_Z', 'Magn_Y', 'Magn_X', 'Acc_Z', 'Acc_Y', 'Acc_X', 'Gyro_Z', 'Gyro_Y', 'Gyro_X', 'Heading', 'Pitch', 'Roll', 'P_Bar', 'Altitude', 'Long', 'Lat']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Create Extended Kalman Filter
num_states = 9  # Number of states to track
dim_z = 3  # Number of measurements

ekf = ExtendedKalmanFilter(dim_x=num_states, dim_z=dim_z)

# Initialize Extended Kalman Filter
initial_state = np.zeros(num_states)
initial_covariance = np.eye(num_states)

ekf.x = initial_state
ekf.P = initial_covariance

H = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0]])

def h(x):
    # Measurement function: relates the state to the measurements
    return np.dot(H, x)

def H_jacobian(x):
    # Jacobian of the measurement function: partial derivatives of h with respect to the state variables
    return H

ekf.H = H
ekf.h = h
ekf.HJacobian = H_jacobian

# Define transition function and process noise
dt = 1.0
def f(x):
    F = np.eye(num_states)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    return F.dot(x)

ekf.f = f
ekf.Q = np.eye(num_states) * 0.01

# Apply Extended Kalman Filter to the dataset
filtered_data = []
for index, row in data.iterrows():
    measurement = np.array([row['Acc_X'], row['Acc_Y'], row['Acc_Z']]).astype(np.float64)
    ekf.predict()
    ekf.update(measurement,H_jacobian,h)
    filtered_data.append(ekf.x[:dim_z])

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
