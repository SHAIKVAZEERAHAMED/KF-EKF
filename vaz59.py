import pandas as pd
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
import matplotlib.pyplot as plt

# File path of the dataset
file_path1 = r"C:\Users\SHAIK VAZEER AHAMED\Downloads\magnetic_data.csv"

# Load the data from CSV file
data1 = pd.read_csv(file_path1)

num_states1 = 5  # Number of states to track (Pos(Mx), Pos(My), Pos(Mz), dis_m, ang_m)
dim_z1 = 5  # Number of measurements (Pos(Mx), Pos(My), Pos(Mz), dis_m, ang_m)

# Extract the relevant columns from the dataset
sno_m = data1['sno_m']
pos_mx = data1['Pos(Mx)']
pos_my = data1['Pos(My)']
pos_mz = data1['Pos(Mz)']
dis_m = data1['dis_m']
ang_m = data1['ang_m']

# Create an instance of the ExtendedKalmanFilter
ekf = ExtendedKalmanFilter(dim_x=num_states1, dim_z=dim_z1)

# Define the state transition function (motion model)
def transition_function(x):
    # Update the state variables based on the motion model
    dt = 1  # Time step (adjust as per your data)
    x[0] += x[3] * np.cos(x[4]) * dt
    x[1] += x[3] * np.sin(x[4]) * dt
    return x

# Define the measurement function
def measurement_function(x):
    # Map the state variables to the measurement space
    # Here, we directly return the measured values
    return x[:dim_z1]

# Define the state transition matrix (F)
ekf.F = np.array([[1, 0, 0, np.cos(ang_m[0]), -dis_m[0] * np.sin(ang_m[0])],
                  [0, 1, 0, np.sin(ang_m[0]), dis_m[0] * np.cos(ang_m[0])],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])

# Set the initial state
initial_state = np.array([pos_mx[0], pos_my[0], pos_mz[0], dis_m[0], ang_m[0]])

# Set the initial covariance matrix
initial_covariance = np.eye(num_states1)

# Set the process noise covariance matrix
process_noise_cov = np.eye(num_states1) * 0.1  # Adjust the values as per your requirements

# Set the measurement noise covariance matrix
measurement_noise_cov = np.eye(dim_z1) * 0.1  # Adjust the values as per your requirements

# Set the initial state and covariance for the Extended Kalman Filter
ekf.x = initial_state
ekf.P = initial_covariance
ekf.Q = process_noise_cov
ekf.R = measurement_noise_cov

# Initialize lists to store the estimated states and measurements
estimated_states = []
measurements = []

# Set the control input matrix (B)
ekf.B = np.array([[0, 0],
                  [0, 0],
                  [0, 0],
                  [1, 0],
                  [0, 0]])

# ...

# Run the Extended Kalman Filter for each data point
for i in range(len(sno_m)):
    # Obtain the current measurement
    measurement = np.array([pos_mx[i], pos_my[i], pos_mz[i], dis_m[i], ang_m[i]])
    measurements.append(measurement)

    # Set the control input for the motion model
    u = np.array([0, 0])  # Replace with appropriate control input values
    ekf.predict(transition_function, u=u)

    # Update the state based on the measurement
    ekf.update(measurement_function, measurement)

    # Store the estimated state
    estimated_states.append(ekf.x)

# ...


# Extract the estimated position values from the estimated states
estimated_pos_mx = [state[0] for state in estimated_states]
estimated_pos_my = [state[1] for state in estimated_states]
estimated_pos_mz = [state[2] for state in estimated_states]

# Plot the estimated positions
plt.plot(sno_m, pos_mx, label='Measured Pos(Mx)')
plt.plot(sno_m, estimated_pos_mx, label='Estimated Pos(Mx)')
plt.plot(sno_m, pos_my, label='Measured Pos(My)')
plt.plot(sno_m, estimated_pos_my, label='Estimated Pos(My)')
plt.plot(sno_m, pos_mz, label='Measured Pos(Mz)')
plt.plot(sno_m, estimated_pos_mz, label='Estimated Pos(Mz)')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Estimated Positions')
plt.legend()
plt.show()
