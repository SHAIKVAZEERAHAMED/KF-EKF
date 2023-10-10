import pandas as pd
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
import matplotlib.pyplot as plt

# File path of the dataset
file_path = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\vision_EM_data_1june2023.csv"

# Load the data from CSV file
data = pd.read_csv(file_path)

# Create Extended Kalman filter
num_states = 4  # Number of states to track (Pos(Mx), Pos(My), ang_m, ang_m)
dim_z = 4  # Number of measurements (Pos(Mx), Pos(My), ang_m, ang_m)

ekf = ExtendedKalmanFilter(dim_x=num_states, dim_z=dim_z)

# Initialize Extended Kalman filter
initial_state = np.array([data['X'].iloc[0], data['Y'].iloc[0], data['dis_v'].iloc[0], data['ang_v'].iloc[0]])
initial_covariance = np.eye(num_states) * 0.1  # Adjust the initial covariance as needed

ekf.x = initial_state
ekf.P = initial_covariance

# Define the state transition function for the Extended Kalman Filter
def state_transition_function(x, dt):
    # Modify the state transition matrix F based on the non-linear dynamics of the system
    # Example: Use a simple constant velocity model with a circular trajectory
    x_next = x.copy()
    x_next[0] += x[2] * np.cos(x[3]) * dt
    x_next[1] += x[2] * np.sin(x[3]) * dt
    return x_next
dt = 1

# Define the predicted measurement function for the Extended Kalman Filter
def predicted_measurement(x):
    # Modify the predicted measurement based on the non-linear relationship between states and measurements
    # Example: Directly use the states as predicted measurements
    return x

# Set the state transition function and predicted measurement for the Extended Kalman Filter
ekf.f = state_transition_function
ekf.Hx = predicted_measurement

# Define the process noise covariance matrix
process_noise = np.eye(num_states) * 0.01  # Adjust the process noise covariance as needed
ekf.Q = process_noise

# Define the measurement noise covariance matrix
measurement_noise = np.eye(dim_z) * 0.1  # Adjust the measurement noise covariance as needed
ekf.R = measurement_noise

# Define the measurement function and its Jacobian
def measurement_function(x):
    return np.array([x[0], x[1], x[2], x[3]])

def measurement_jacobian(x):
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

ekf.h = measurement_function
ekf.H = measurement_jacobian

# Apply Extended Kalman Filter to estimate the true values
estimated_values = []
for _, row in data.iterrows():
    measurement = np.array([
        row['X'],
        row['Y'],
        row['dis_v'],
        row['ang_v']
    ])

    # Predict the object's state
    ekf.predict()

    # Update the object's state based on the measurement
    ekf.update(measurement,measurement_jacobian,measurement_function)

    # Save the estimated state
    estimated_values.append(ekf.x.copy())

estimated_values = np.array(estimated_values)

# Plot the original and estimated values
plt.figure()
plt.plot(data['X'], data['Y'], 'g*-', label='Original')
plt.plot(estimated_values[:, 0], estimated_values[:, 1], 'y*-', label='Estimated')

plt.xlabel('Position X')
plt.ylabel('Position Y')
plt.title('Original and Estimated Positions')
plt.legend()

plt.figure()
plt.plot(data['sno_v'], data['dis_v'], 'yo-', label='Original dis_m')
plt.plot(data['sno_v'], estimated_values[:, 2], 'r*-', label='Estimated dis_m')

plt.xlabel('Sample Number')
plt.ylabel('dis_m')
plt.title('Original and Estimated Distance')
plt.legend()

plt.figure()
plt.plot(data['sno_v'], data['ang_v'], 'mo-', label='Original ang_m')
plt.plot(data['sno_v'], estimated_values[:, 3], 'r*-', label='Estimated ang_m')
plt.xlabel('Sample Number')
plt.ylabel('ang_m')
plt.title('Original and Estimated Angle')
plt.legend()
plt.show()

# Calculate average differences in X and Y
avg_x = np.mean(estimated_values[:, 0] - data['X'])
avg_y = np.mean(estimated_values[:, 1] - data['Y'])

print("Average difference in X:", avg_x)
print("Average difference in Y:", avg_y)
