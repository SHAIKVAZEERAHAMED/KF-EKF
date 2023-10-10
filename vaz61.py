import pandas as pd
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
import matplotlib.pyplot as plt

# File path of the dataset
file_path = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\M64_1_S_N.csv"

# Load the data from CSV file
data = pd.read_csv(file_path)
data['M64_1_S_N'] = data['M64_1_S_N'].astype(float)

# Create Extended Kalman Filter
num_states = 1  # Number of states to track
dim_z = 1  # Number of measurements

ekf = ExtendedKalmanFilter(dim_x=num_states, dim_z=dim_z)

# Initialize Extended Kalman Filter
initial_state = np.array([data['M64_1_S_N'].iloc[0]])
initial_covariance = np.eye(num_states) * 0.1  # Adjust the initial covariance as needed

ekf.x = initial_state[:, np.newaxis]  # Add np.newaxis to convert 1D array to 2D column vector
ekf.P = initial_covariance

# Define the state transition function for EKF
def state_transition_function(x, dt):
    # Modify this function based on your system dynamics
    # Here's an example assuming a nonlinear system
    x1 = x[0]**2 + np.sin(x[0])
    return np.array([x1])

# Define the Jacobian of the state transition function for EKF
def state_transition_jacobian(x, dt):
    # Modify this function to calculate the Jacobian matrix of the state transition function
    # Here's an example assuming a nonlinear system
    jacobian = 2 * x[0] + np.cos(x[0])
    return np.array([[jacobian]])

# Define the measurement function for EKF
def measurement_function(x):
    # Modify this function based on your measurement model
    # Here's an example assuming a direct measurement of the state
    return np.array([x[0]])

# Define the Jacobian of the measurement function for EKF
def measurement_jacobian(x):
    # Modify this function to calculate the Jacobian matrix of the measurement function
    # Here's an example assuming a direct measurement of the state
    return np.array([[1]])

# Apply Extended Kalman Filter to estimate the true values
ekf_estimated_values = []
for _, row in data.iterrows():
    measurement = np.array([row['M64_1_S_N']])

    # Set the state transition and measurement functions in the EKF
    ekf.f = state_transition_function
    ekf.F = state_transition_jacobian
    ekf.h = measurement_function
    ekf.H = measurement_jacobian

    # Predict the state using the state transition function and its Jacobian matrix
    predicted_state = state_transition_function(ekf.x.flatten(), dt=1)
    predicted_covariance = state_transition_jacobian(ekf.x.flatten(), dt=1) @ ekf.P @ state_transition_jacobian(ekf.x.flatten(), dt=1).T

    # Update the state and covariance
    ekf.x = predicted_state[:, np.newaxis]
    ekf.P = predicted_covariance

    # Update using the measurement
    ekf.update(measurement,measurement_jacobian,measurement_function)
    ekf_estimated_values.append(ekf.x.copy())

ekf_estimated_values = np.array(ekf_estimated_values)

plt.figure()
# Plot the original and EKF-estimated values
plt.plot(data['M64_1_S_N'], 'g*-', label='Original')
plt.plot(ekf_estimated_values[:, 0], 'r*-', label='EKF Estimated')

plt.xlabel('Sample')
plt.ylabel('S/N for M64 channel 1')
plt.title('Original and EKF-Estimated')
plt.legend()
plt.show()