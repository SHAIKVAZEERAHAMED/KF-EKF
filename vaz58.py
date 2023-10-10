import pandas as pd
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
import matplotlib.pyplot as plt

# File path of the dataset
file_path = r"C:\Users\SHAIK VAZEER AHAMED\Downloads\magnetic_data.csv"

# Load the data from CSV file
data = pd.read_csv(file_path)

# Create Extended Kalman Filter
num_states = 4  # Number of states to track (Pos(Mx), Pos(My), Pos(Mz), dis_m, ang_m)
dim_z = 4  # Number of measurements (Pos(Mx), Pos(My), Pos(Mz), dis_m, ang_m)

ekf = ExtendedKalmanFilter(dim_x=num_states, dim_z=dim_z)

# Initialize Extended Kalman Filter
initial_state = np.array([data['Pos(Mx)'].iloc[0], data['Pos(My)'].iloc[0], data['dis_m'].iloc[0], data['ang_m'].iloc[0]])
initial_covariance = np.eye(num_states) * 0.1  # Adjust the initial covariance as needed

ekf.x = initial_state
ekf.P = initial_covariance

# Define the state transition function with nonlinearity
def state_transition_function(x, dt):
    # Nonlinear state transition function
    x_pred = np.zeros_like(x)
    x_pred[0] = x[0] + dt * x[1]  # Nonlinear propagation for Pos(Mx)
    x_pred[1] = x[1] * np.sin(x[0])  # Nonlinear propagation for Pos(My)
    x_pred[2] = x[2] + dt * x[3]  # Nonlinear propagation for dis_m
    x_pred[3] = x[3]  # No change for ang_m
    return x_pred

# Define the measurement function
def measurement_function(x):
    pos_mx = x[0]
    pos_my = x[1]
    dis_m = x[2]
    ang_m = x[3]

    measurement = np.array([pos_mx, pos_my, dis_m, ang_m])
    return measurement

# Define the measurement Jacobian function
def measurement_jacobian(x):
    # The Jacobian matrix will be the identity matrix since the measurement function is linear
    jacobian = np.eye(dim_z, num_states)  # Identity matrix as the Jacobian
    return jacobian

# Set the state transition, measurement, and measurement Jacobian functions
ekf.f = state_transition_function
ekf.h = measurement_function
ekf.HJacobian = measurement_jacobian

# Define the process noise covariance matrix
process_noise = np.eye(num_states) * 0.01  # Adjust the process noise covariance as needed
ekf.Q = process_noise

# Define the measurement noise covariance matrix
measurement_noise = np.eye(dim_z) * 0.1  # Adjust the measurement noise covariance as needed
ekf.R = measurement_noise

# ...

# Set the state transition, measurement, and measurement Jacobian functions
ekf.f = state_transition_function
ekf.h = measurement_function
ekf.HJacobian = measurement_jacobian

# Apply Extended Kalman Filter to estimate the true values
estimated_values = []
for _, row in data.iterrows():
    measurement = np.array([
        row['Pos(Mx)'],
        row['Pos(My)'],
        row['dis_m'],
        row['ang_m']
    ])

    # Predict the object's state
    ekf.predict()

    # Update the object's state based on the measurement
    ekf.update(z=measurement, HJacobian=ekf.HJacobian, Hx=ekf.h)

    # Save the estimated state
    estimated_values.append(ekf.x.copy())

estimated_values = np.array(estimated_values)

# Plot the original and estimated values
plt.figure()
plt.plot(data['Pos(Mx)'], data['Pos(My)'], 'g*-', label='Original')
plt.plot(estimated_values[:, 0], estimated_values[:, 1], 'yo-', label='Estimated')

plt.xlabel('Position (Mx)')
plt.ylabel('Position (My)')
plt.title('Original and Estimated Positions')
plt.legend()

plt.figure()
plt.plot(data['sno_m'], data['dis_m'], 'yo-', label='Original dis_m')
plt.plot(data['sno_m'], estimated_values[:, 2], 'ro-', label='Estimated dis_m')

plt.xlabel('Sample Number')
plt.ylabel('dis_m ')
plt.title('Original and Estimated Distance')
plt.legend()

plt.figure()
plt.plot(data['sno_m'], data['ang_m'], 'mo-', label='Original ang_m')
plt.plot(data['sno_m'], estimated_values[:, 3], 'ro-', label='Estimated ang_m')

plt.xlabel('Sample Number')
plt.ylabel('ang_m')
plt.title('Original and Estimated Angle')
plt.legend()

plt.show()
# diff_x = []
# diff_y = []

# for i in range(len(data['Pos(Mx)'])):
#     diff_x.append(estimated_values[:, 0] - data['Pos(Mx)'][i])
#     diff_y.append(estimated_values[:, 1] - data['Pos(My)'][i])

# avg_x = sum(estimated_values[:, 0] - data['Pos(Mx)']) / len(data['Pos(Mx)'])
# avg_y = sum(estimated_values[:, 1] - data['Pos(My)']) / len(data['Pos(Mx)'])

# print("Average difference in X:", avg_x)
# print("Average difference in Y:", avg_y)

