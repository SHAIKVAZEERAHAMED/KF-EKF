import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load radar data
radar_data1 = pd.read_csv(r'C:\Users\SHAIK VAZEER AHAMED\Downloads\philos_2019_10_29__16_00_19_target_and_sailboats\philos_2019_10_29__16_00_19_target_and_sailboats\csv\radar_0_segments.csv')
radar_data2 = pd.read_csv(r'C:\Users\SHAIK VAZEER AHAMED\Downloads\philos_2019_10_29__16_00_19_target_and_sailboats\philos_2019_10_29__16_00_19_target_and_sailboats\csv\radar_1_segments.csv')

# Remove leading and trailing spaces from column names
radar_data1.columns = radar_data1.columns.str.strip()
radar_data2.columns = radar_data2.columns.str.strip()

# Extract relevant columns from radar data
radar_timestamps1 = radar_data1['time']
radar_angles1 = radar_data1['angle']
radar_intensities1 = radar_data1.iloc[:, 4:]  # Assuming intensity columns start from index 4

radar_timestamps2 = radar_data2['time']
radar_angles2 = radar_data2['angle']
radar_intensities2 = radar_data2.iloc[:, 4:]  # Assuming intensity columns start from index 4


# Plotting the results
plt.figure()

# Plot object trajectory before filtering - Radar 1
plt.subplot(2, 2, 1)
plt.plot(radar_timestamps1, radar_angles1, label='Angle (Radar 1)')
plt.xlabel('Timestamp')
plt.ylabel('Angle')
plt.legend()
plt.title('Angle - Before Filtering (Radar 1)')

# Plot object trajectory before filtering - Radar 2
plt.subplot(2, 2, 2)
plt.plot(radar_timestamps2, radar_angles2, label='Angle (Radar 2)')
plt.xlabel('Timestamp')
plt.ylabel('Angle')
plt.legend()
plt.title('Angle - Before Filtering (Radar 2)')

# Plot filtered intensities for radar 1
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(radar_timestamps1, radar_intensities1, label='Filtered Intensity (Radar 1)')
plt.xlabel('Timestamp')
plt.ylabel('Intensity')
plt.legend()
plt.title('Estimated Intensity - Radar 1')

# Plot filtered intensities for radar 2
plt.subplot(2, 1, 2)
plt.plot(radar_timestamps2, radar_intensities2, label='Filtered Intensity (Radar 2)')
plt.xlabel('Timestamp')
plt.ylabel('Intensity')
plt.legend()
plt.title('Estimated Intensity - Radar 2')
plt.tight_layout()
plt.show()

# Kalman filter initialization
dt = 1.0  # Time step

# State transition matrix
A = np.array([[1, dt], [0, 1]])

# Observation matrix
H = np.array([[1, 0]])

# Measurement noise covariance matrix
R = np.array([[1]])

# Process noise covariance matrix
Q = np.array([[0.001, 0], [0, 0.001]])

# Initial state vector
x = np.array([[radar_angles1[0]], [0]])

# Initial state covariance matrix
P = np.eye(2)

# Initialize filtered lists
filtered_angles1 = []
filtered_intensities1 = []

# Kalman filter loop for radar 1
for i in range(len(radar_data1)):
    # Predict
    x = np.dot(A, x)
    P = np.dot(np.dot(A, P), A.T) + Q

    # Update - using angle and intensity
    y_angle = radar_angles1[i] - np.dot(H, x)
    y_intensity = radar_intensities1.iloc[i, :].values - np.dot(H, x)

    S = np.dot(np.dot(H, P), H.T) + R
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    x = x + np.dot(K, y_angle)
    P = np.dot((np.eye(2) - np.dot(K, H)), P)

    filtered_angles1.append(x[0][0])
    filtered_intensities1.append(x[1][0])

# Initialize filtered lists
filtered_angles2 = []
filtered_intensities2 = []

# Kalman filter loop for radar 2
for i in range(len(radar_data2)):
    # Predict
    x = np.dot(A, x)
    P = np.dot(np.dot(A, P), A.T) + Q

    # Update - using angle and intensity
    y_angle = radar_angles2[i] - np.dot(H, x)
    y_intensity = radar_intensities2.iloc[i, :].values - np.dot(H, x)
    S = np.dot(np.dot(H, P), H.T) + R
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    x = x + np.dot(K, y_angle)
    P = np.dot((np.eye(2) - np.dot(K, H)), P)

    filtered_angles2.append(x[0][0])
    filtered_intensities2.append(x[1][0])

# Plot filtered angles for radar 1
plt.subplot(2, 2, 3)
plt.plot(radar_timestamps1, filtered_angles1, label='Filtered Angle (Radar 1)')
plt.xlabel('Timestamp')
plt.ylabel('Angle')
plt.legend()
plt.title('Estimated Angle - Radar 1')

# Plot filtered angles for radar 2
plt.subplot(2, 2, 4)
plt.plot(radar_timestamps2, filtered_angles2, label='Filtered Angle (Radar 2)')
plt.xlabel('Timestamp')
plt.ylabel('Angle')
plt.legend()
plt.title('Estimated Angle - Radar 2')

# Plot filtered intensities for radar 1
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(radar_timestamps1, filtered_intensities1, label='Filtered Intensity (Radar 1)')
plt.xlabel('Timestamp')
plt.ylabel('Intensity')
plt.legend()
plt.title('Estimated Intensity - Radar 1')

# Plot filtered intensities for radar 2
plt.subplot(2, 1, 2)
plt.plot(radar_timestamps2, filtered_intensities2, label='Filtered Intensity (Radar 2)')
plt.xlabel('Timestamp')
plt.ylabel('Intensity')
plt.legend()
plt.title('Estimated Intensity - Radar 2')

plt.tight_layout()
plt.show()
