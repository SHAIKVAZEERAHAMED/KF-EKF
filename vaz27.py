import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV files
path1_data = pd.read_csv(r'C:\Users\SHAIK VAZEER AHAMED\Downloads\philos_2019_10_29__16_00_19_target_and_sailboats\philos_2019_10_29__16_00_19_target_and_sailboats\csv\radar_0_segments.csv')
path2_data = pd.read_csv(r'C:\Users\SHAIK VAZEER AHAMED\Downloads\philos_2019_10_29__16_00_19_target_and_sailboats\philos_2019_10_29__16_00_19_target_and_sailboats\csv\radar_1_segments.csv')

# Extract the relevant columns from each dataset
time_path1 = path1_data['time']
angle_measurements_path1 = path1_data['angle']

time_path2 = path2_data['time']
angle_measurements_path2 = path2_data['angle']

# Combine the angle measurements from both paths
time = pd.concat([time_path1, time_path2])
angle_measurements = pd.concat([angle_measurements_path1, angle_measurements_path2])

def kalman_filter_update(state_estimate, covariance_matrix, measurement, measurement_noise):
    # Update
    innovation = measurement - measurement_matrix @ state_estimate
    innovation_covariance = measurement_matrix @ covariance_matrix @ measurement_matrix.T + measurement_noise
    kalman_gain = covariance_matrix @ measurement_matrix.T / innovation_covariance
    
    updated_state = state_estimate + kalman_gain * innovation
    updated_covariance = (np.eye(state_estimate.shape[0]) - kalman_gain @ measurement_matrix) @ covariance_matrix

    return updated_state, updated_covariance


# Initialize the state estimate and covariance matrix
state_estimate = np.zeros((1, 1))  # Assuming a 1-dimensional state vector (angle)
covariance_matrix = np.eye(1)  # Assuming an identity covariance matrix for the angle

# Define the measurement matrix and noise covariance matrix
measurement_matrix = np.array([[1]])  # Assuming direct measurement of the angle variable
measurement_noise_covariance = 0.1  # Increase the value if needed

# Determine the length of the shorter path
min_length = min(len(angle_measurements_path1), len(angle_measurements_path2))

# Perform the Kalman filter update for each measurement up to the length of the shorter path
estimated_states = []
for i in range(min_length):
    measurement = np.array([angle_measurements_path1.iloc[i], angle_measurements_path2.iloc[i]])
    state_estimate, covariance_matrix = kalman_filter_update(state_estimate, covariance_matrix, measurement, measurement_noise_covariance)
    estimated_states.append(state_estimate.flatten())

# Extract the estimated angles from the estimated states
estimated_angles = np.array(estimated_states)


# # Plot the estimated angles from radar 1 and radar 2
# plt.plot(time_path1[:min_length], angle_measurements_path1[:min_length], marker='o', label='Radar 1')
# plt.plot(time_path2[:min_length], angle_measurements_path2[:min_length], label='Radar 2')
# plt.xlabel('Time')
# plt.ylabel('Angle')
# plt.title('Estimated Object Angles')
# plt.legend()
# plt.show()
# Split the estimated angles for each path
estimated_angles_path1 = estimated_angles[:, 0]
estimated_angles_path2 = estimated_angles[:, 1]


# Plot the estimated object trajectory for path 1
plt.figure()
plt.plot(estimated_angles_path1, 'bo-', label='Object Trajectory (Path 1)')
plt.legend()

plt.plot(estimated_angles_path2, 'r', label='Object Trajectory (Path 2)')
plt.xlabel('Time')
plt.ylabel('Angle')
plt.title('Object Trajectory')
plt.legend()
plt.show()

