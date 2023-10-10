import pandas as pd
import numpy as np
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
original_acc = data[['Acc_X', 'Acc_Y', 'Acc_Z']].values
original_gyro = data[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].values
original_lat = data['Lat'].values
original_long = data['Long'].values

# Create Extended Kalman Filter
num_states = 8  # Number of states to track (Acc_X, Acc_Y, Acc_Z, Long, Lat, Gyro_X, Gyro_Y, Gyro_Z)
dim_z = 8  # Number of measurements (Acc_X, Acc_Y, Acc_Z, Long, Lat, Gyro_X, Gyro_Y, Gyro_Z)

ekf = ExtendedKalmanFilter(dim_x=num_states, dim_z=dim_z)

# Initialize Extended Kalman Filter
initial_state = data.iloc[0][columns].values.astype(np.float64)
initial_covariance = np.eye(num_states) * 0.1

ekf.x = initial_state
ekf.P = initial_covariance

# Define the state transition function
def f(x, dt):
    F = np.eye(num_states)
    # Update the state transition matrix F using dt
    F[0, 3] = dt  # Update Long based on the time step
    F[1, 4] = dt  # Update Lat based on the time step
    # Update other elements of F if needed
    return F.dot(x)


ekf.f = f

def h(x):
    # Compute the expected measurement based on the current state
    # You need to define the appropriate mapping based on your problem
    acc_x, acc_y, acc_z, long, lat, gyro_x, gyro_y, gyro_z = x
    expected_measurement = np.array([acc_x, acc_y, acc_z, long, lat, gyro_x, gyro_y, gyro_z])
    return expected_measurement

def HJacobian(x):
    # Compute the Jacobian matrix that represents the linearized mapping
    # between the state variables and the measurements
    H = np.eye(dim_z)
    return H


ekf.h = h

ekf.HJacobian = HJacobian

# Define the measurement function
def Hx(x):
    return x

ekf.Hx = Hx

# Define the measurement noise covariance
ekf.R = np.eye(dim_z) * 0.1  # Adjust the value to control the filtering of measurement noise

# Track the object using the Extended Kalman Filter
filtered_states = []
dt = 1.0  # Time step, adjust as needed
for _, row in data.iterrows():
    measurement = row[columns].values.astype(np.float64)
    
    # Predict the object's state
    ekf.predict()
    
    # Update the state transition matrix F using dt
    ekf.F = f(ekf.x, dt)

    # Update the object's state based on the measurement
    ekf.update(measurement, HJacobian, Hx)

    # Save the estimated state
    filtered_states.append(ekf.x.copy())


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
