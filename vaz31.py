import pandas as pd
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
import matplotlib.pyplot as plt

# File path of the dataset
file_path = r"C:\Users\SHAIK VAZEER AHAMED\Downloads\magnetic_data_jm - -30.csv"

# Load the data from CSV file
data = pd.read_csv(file_path)

# Create Extended Kalman filter
num_states = 4  # Number of states to track (Pos(Mx), Pos(My) dis_m, ang_m)
dim_z = 4  # Number of measurements (Pos(Mx), Pos(My) dis_m, ang_m)

ekf = ExtendedKalmanFilter(dim_x=num_states, dim_z=dim_z)

# Initialize Extended Kalman filter
initial_state = np.array([data['Pos(Mx)'].iloc[0], data['Pos(My)'].iloc[0], data['dis_m'].iloc[0], data['ang_m'].iloc[0]])
initial_covariance = np.eye(num_states) * 0.1  # Adjust the initial covariance as needed

ekf.x = initial_state
ekf.P = initial_covariance

# Define the state transition function
def state_transition_function(x, dt):
    # Extract the state variables
    pos_x, pos_y, dis_m, ang_m = x

    # Compute the updated state variables
    new_pos_x = pos_x + dt * pos_y
    new_pos_y = pos_y  # Assume no change in pos_y
    new_dis_m = dis_m  # Assume no change in dis_m
    new_ang_m = ang_m  # Assume no change in ang_m

    # Return the updated state
    return np.array([new_pos_x, new_pos_y, new_dis_m, new_ang_m])

# Define the measurement function
def measurement_function(x):
    return x

# Set the state transition and measurement functions
ekf.f = state_transition_function
ekf.h = measurement_function


# Define the Jacobian matrix of the measurement function
def measurement_jacobian(x):
    # Extract the state variables
    pos_x, pos_y, dis_m, ang_m = x

    dh_dx = np.eye(dim_z)

    # Return the Jacobian matrix
    return dh_dx

ekf.H = measurement_jacobian

# Apply Extended Kalman filter to estimate the true values
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
    ekf.update(measurement, ekf.H, ekf.h)

    # Save the estimated state
    estimated_values.append(ekf.x.copy())

estimated_values = np.array(estimated_values)

# Plot the original and estimated values
plt.figure()
plt.plot(data['Pos(Mx)'], data['Pos(My)'], 'g*-', label='Original')
plt.plot(estimated_values[:, 0], estimated_values[:, 1], 'y*-', label='Estimated')

plt.xlabel('Position (Mx)')
plt.ylabel('Position (My)')
plt.title('Original and Estimated Positions')
plt.legend()

plt.figure()
plt.plot(data['sno_m'], data['dis_m'], 'yo-', label='Original dis_m')
plt.plot(data['sno_m'], estimated_values[:, 2], 'r*-', label='Estimated dis_m')

plt.xlabel('Sample Number')
plt.ylabel('dis_m')
plt.title('Extended Kalman Filter - Estimated Distance')
plt.legend()

plt.figure()
plt.plot(data['sno_m'], data['ang_m'], 'mo-', label='Original ang_m')
plt.plot(data['sno_m'], estimated_values[:, 3], 'r*-', label='Estimated ang_m')
plt.xlabel('Sample Number')
plt.ylabel('ang_m')
plt.title('Extended Kalman Filter - Estimated Angle')
plt.legend()

plt.show()
