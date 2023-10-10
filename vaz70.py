import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt

# Load the magnetic data and vision data from CSV files
magnetic_data = pd.read_csv(r"C:\Users\SHAIK VAZEER AHAMED\Downloads\magnetic_data.csv")
vision_data = pd.read_csv(r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\vision_EM_data_1june2023.csv")

# Extract the relevant columns from the magnetic data
magnetic_measurements = magnetic_data[['Pos(Mx)', 'Pos(My)', 'dis_m', 'ang_m']].values

# Extract the relevant columns from the vision data and add the magnetic measurements
vision_measurements = vision_data[['dis_v', 'ang_v', 'X', 'Y']].values

# Initialize the Kalman filter
kalman_filter = KalmanFilter(dim_x=8, dim_z=4)

# Define the initial state and covariance matrix
initial_state = np.zeros(8)  # Modify this according to your system
initial_covariance = np.eye(8)  # Modify this according to your system

# Set the initial state and covariance matrix of the Kalman filter
kalman_filter.x = initial_state
kalman_filter.P = initial_covariance

# Define the measurement matrix
measurement_matrix = np.zeros((4, 8))
measurement_matrix[0, 0] = 1  # dis_m
measurement_matrix[1, 1] = 1  # ang_m
measurement_matrix[2, 2] = 1  # X
measurement_matrix[3, 3] = 1  # Y
kalman_filter.H = measurement_matrix

# Define the measurement noise covariance matrix
measurement_noise_cov = np.eye(4)  # Modify this according to your system
kalman_filter.R = measurement_noise_cov

# Define the process noise covariance matrix
process_noise_cov = np.eye(8)  # Modify this according to your system
kalman_filter.Q = process_noise_cov

# Create empty lists to store the filtered states and fused measurements
filtered_states = []
fused_measurements = []

# Iterate over the measurements
for measurement in vision_measurements:
    measurement = measurement.reshape((4, 1))  # Reshape the measurement to (4, 1)
    
    # Perform the measurement update step
    kalman_filter.update(measurement)
    
    # Perform the state prediction step
    kalman_filter.predict()
    
    # Append the filtered state and fused measurement
    filtered_states.append(kalman_filter.x)
    fused_measurement = np.dot(kalman_filter.H, kalman_filter.x)
    fused_measurements.append(fused_measurement)

# Convert the filtered states and fused measurements into DataFrames
filtered_states_df = pd.DataFrame(filtered_states, columns=['filtered_dis_m', 'filtered_ang_m', 'filtered_X', 'filtered_Y', 'filtered_Pos(Mx)', 'filtered_Pos(My)', 'filtered_dis_m2', 'filtered_ang_m2'])
fused_measurements_df = pd.DataFrame(fused_measurements, columns=['fused_dis_m', 'fused_ang_m', 'fused_X', 'fused_Y'])

# Save the filtered states and fused measurements to CSV files
filtered_states_df.to_csv('filtered_states.csv', index=False)
fused_measurements_df.to_csv('fused_measurements.csv', index=False)

# Plot the fused measurements and filtered states
plt.figure(figsize=(12, 8))

# Plot the fused distance measurements
plt.subplot(2, 2, 1)
plt.plot(fused_measurements_df['fused_dis_m'], label='Fused Distance')
plt.xlabel('Time')
plt.ylabel('Distance')
plt.legend()

# Plot the fused angle measurements
plt.subplot(2, 2, 2)
plt.plot(fused_measurements_df['fused_ang_m'], label='Fused Angle')
plt.xlabel('Time')
plt.ylabel('Angle')
plt.legend()

# Plot the fused Pos(Mx) and X measurements
plt.subplot(2, 2, 3)
plt.plot(fused_measurements_df['fused_X'], label='Fused X')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()

# Plot the fused Pos(My) and Y measurements
plt.subplot(2, 2, 4)
plt.plot(fused_measurements_df['fused_Y'], label='Fused Y')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()

plt.tight_layout()
plt.show()
