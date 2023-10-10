# RxPower M64
import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter
import matplotlib.pyplot as plt

# File path of the dataset
file_path = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\M64_6_S_N.csv"

# Load the data from CSV file
data = pd.read_csv(file_path)
data['M64_6_S_N'] = data['M64_6_S_N'].astype(float)

# Create Kalman Filter
num_states = 1  # Number of states to track
dim_z = 1  # Number of measurements

kf = KalmanFilter(dim_x=num_states, dim_z=dim_z)

# Initialize Kalman Filter
initial_state = np.array([data['M64_6_S_N'].iloc[0]])
initial_covariance = np.eye(num_states) * 0.1  # Adjust the initial covariance as needed

kf.x = initial_state[:, np.newaxis]  # Add np.newaxis to convert 1D array to 2D column vector
kf.P = initial_covariance

# Define the state transition matrix
dt = 1  # Time step
kf.F = np.array([[1]])

# Define the measurement matrix
kf.H = np.eye(dim_z)

# Define the process noise covariance matrix
process_noise = np.eye(num_states) * 0.01  # Adjust the process noise covariance as needed
kf.Q = process_noise

# Define the measurement noise covariance matrix
measurement_noise = np.eye(dim_z) * 0.1  # Adjust the measurement noise covariance as needed
kf.R = measurement_noise

# Create Extended Kalman Filter
ekf = ExtendedKalmanFilter(dim_x=num_states, dim_z=dim_z)

# Initialize Extended Kalman Filter
ekf.x = initial_state[:, np.newaxis]  # Add np.newaxis to convert 1D array to 2D column vector
ekf.P = initial_covariance

# Define the state transition function for EKF
def state_transition_function(x, dt):
    
    F = np.array([[1, dt],
                  [0, 1]])
    return np.dot(F, x)

# Define the measurement function for EKF
def measurement_function(x):
    
    return np.array([x[0]])

# Define the Jacobian of the measurement function for EKF
def measurement_jacobian(x):
    return np.array([[1]])

# Set the state transition, measurement, and measurement Jacobian functions in the EKF
ekf.f = state_transition_function
ekf.h = measurement_function
ekf.H = measurement_jacobian

# Define the process noise covariance matrix for EKF
ekf.Q = process_noise

# Define the measurement noise covariance matrix for EKF
ekf.R = measurement_noise

# Apply Kalman Filter and Extended Kalman Filter to estimate the true values
kf_estimated_values = []
ekf_estimated_values = []
for _, row in data.iterrows():
    measurement = np.array([row['M64_6_S_N']])

    # Predict and update using Kalman Filter
    kf.predict()
    kf.update(measurement)
    kf_estimated_values.append(kf.x.copy())

    # Predict and update using Extended Kalman Filter
    ekf.predict()
    ekf.update(measurement, measurement_jacobian, measurement_function)
    ekf_estimated_values.append(ekf.x.copy())

kf_estimated_values = np.array(kf_estimated_values)
ekf_estimated_values = np.array(ekf_estimated_values)

plt.figure()
# Plot the original, KF-estimated, and EKF-estimated values
plt.plot(data['M64_6_S_N'], 'g*-', label='Original')
plt.plot(kf_estimated_values[:, 0], 'bo-', label='KF Estimated')
plt.plot(ekf_estimated_values[:, 0], 'r*-', label='EKF Estimated')
    
plt.xlabel('Sample')
plt.ylabel('S/N for M64 channel 6')
plt.title('Original, KF-Estimated, and EKF-Estimated')
plt.legend()
plt.show()
