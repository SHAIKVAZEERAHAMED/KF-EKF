import numpy as np
import pandas as pd
from filterpy.kalman import ExtendedKalmanFilter
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


# Create Extended Kalman filter
num_states = 8  # Number of states to track (Acc_X, Acc_Y, Acc_Z, Long, Lat, Gyro_X, Gyro_Y, Gyro_Z)
dim_z = 8  # Number of measurements (Acc_X, Acc_Y, Acc_Z, Long, Lat, Gyro_X, Gyro_Y, Gyro_Z)

ekf = ExtendedKalmanFilter(dim_x=num_states, dim_z=dim_z)


# Initialize Extended Kalman filter
initial_state = data.iloc[0][columns].values.astype(np.float64)
initial_covariance = np.eye(num_states) * 0.1

ekf.x = initial_state
ekf.P = initial_covariance


# Define state transition matrix and control input matrix
dt = 1.0
ekf.F = np.eye(num_states)
ekf.B = np.eye(num_states)


# Define measurement function
def h(x):
    # Modify this function based on the measurements available in your system
    return x


# Define the Jacobian matrix of the measurement function
def H(x):
    # Modify this function based on the Jacobian of your system
    return np.eye(dim_z, num_states)


ekf.h = h
ekf.H = H


# Define process noise covariance and measurement noise covariance
ekf.Q = np.eye(num_states) * 0.01
ekf.R = np.eye(dim_z) * 0.1  # Adjust the value to control the filtering of measurement noise


# Extract original data for plotting
original_acc = data[['Acc_X', 'Acc_Y', 'Acc_Z']].values
original_gyro = data[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].values
original_lat = data['Lat'].values
original_long = data['Long'].values

# Define the Jacobian matrix of the measurement function
def HJacobian(x):
    # Modify this function based on the Jacobian of your system
    return np.eye(dim_z, num_states)


# Define the predicted measurement function
def Hx(x):
    # Modify this function based on the predicted measurement equations of your system
    return x


# Set the HJacobian and Hx functions in the ExtendedKalmanFilter object
ekf.HJacobian = HJacobian
ekf.Hx = Hx

# Track the object using the Extended Kalman filter
filtered_states = []
for _, row in data.iterrows():
    measurement = row[columns].values.astype(np.float64)
    ekf.predict()
    ekf.update(measurement,HJacobian,Hx)
    filtered_states.append(ekf.x.copy())


# Convert filtered states to numpy array
filtered_states = np.array(filtered_states)


# Extract filtered data for plotting
filtered_acc = filtered_states[:, :3]
filtered_gyro = filtered_states[:, 5:8]
filtered_lat = filtered_states[:,0]
filtered_long = filtered_states[:,0]


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
plt.legend(loc='upper right')


# Plot original gyroscope
plt.subplot(2, 4, 2)
plt.plot(original_gyro[:, 0], label='Original Gyro_X')
plt.plot(original_gyro[:, 1], label='Original Gyro_Y')
plt.plot(original_gyro[:, 2], label='Original Gyro_Z')
plt.xlabel('Index')
plt.ylabel('Angular Velocity')
plt.title('Original Gyro')
plt.legend(loc='upper right')


# Plot original latitude
plt.subplot(2, 4, 3)
plt.plot(original_lat, label='Original Latitude')
plt.xlabel('Index')
plt.ylabel('Latitude')
plt.title('Original Latitude')
plt.legend(loc='upper right')


# Plot original longitude
plt.subplot(2, 4, 4)
plt.plot(original_long, label='Original Longitude')
plt.xlabel('Index')
plt.ylabel('Longitude')
plt.title('Original Longitude')
plt.legend(loc='upper right')


# Plot filtered acceleration
plt.subplot(2, 4, 5)
plt.plot(filtered_acc[:, 0], label='Filtered Acc_X')
plt.plot(filtered_acc[:, 1], label='Filtered Acc_Y')
plt.plot(filtered_acc[:, 2], label='Filtered Acc_Z')
plt.xlabel('Index')
plt.ylabel('Acceleration')
plt.title('Filtered Acceleration')
plt.legend(loc='upper right')


# Plot filtered gyroscope
plt.subplot(2, 4, 6)
plt.plot(filtered_gyro[:, 0], label='Filtered Gyro_X')
plt.plot(filtered_gyro[:, 1], label='Filtered Gyro_Y')
plt.plot(filtered_gyro[:, 2], label='Filtered Gyro_Z')
plt.xlabel('Index')
plt.ylabel('Angular Velocity')
plt.title('Filtered Gyro')
plt.legend(loc='upper right')


# Plot filtered latitude
plt.subplot(2, 4, 7)
plt.plot(filtered_lat, label='Filtered Latitude')
plt.xlabel('Index')
plt.ylabel('Latitude')
plt.title('Filtered Latitude')
plt.legend(loc='upper right')


# Plot filtered longitude
plt.subplot(2, 4, 8)
plt.plot(filtered_long, label='Filtered Longitude')
plt.xlabel('Index')
plt.ylabel('Longitude')
plt.title('Filtered Longitude')
plt.legend(loc='upper right')


plt.tight_layout()
plt.show()

