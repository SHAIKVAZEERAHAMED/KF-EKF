import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV files
csv_file1 = r'C:\Users\SHAIK VAZEER AHAMED\Downloads\philos_2019_10_29__16_00_19_target_and_sailboats\philos_2019_10_29__16_00_19_target_and_sailboats\csv\radar_0_segments.csv'
csv_file2 = r'C:\Users\SHAIK VAZEER AHAMED\Downloads\philos_2019_10_29__16_00_19_target_and_sailboats\philos_2019_10_29__16_00_19_target_and_sailboats\csv\radar_1_segments.csv'

data1 = pd.read_csv(csv_file1, delimiter='\t', header=None)
data2 = pd.read_csv(csv_file2, delimiter='\t', header=None)

# Apply Kalman filter
def kalman_filter(data):
    n = len(data)
    
    # Initialize Kalman filter parameters
    # State vector [position, velocity]
    x = np.array([[data[0]], [0]])
    
    # State transition matrix
    F = np.array([[1, 1], [0, 1]])
    
    # Observation matrix
    H = np.array([[1, 0]])
    
    # Measurement noise covariance matrix
    R = np.array([[1]])
    
    # Process noise covariance matrix
    Q = np.array([[0.001, 0], [0, 0.001]])
    
    # Initial state covariance matrix
    P = np.eye(2)
    
    filtered_data = []
    
    for i in range(n):
        # Predict
        x = np.dot(F, x)
        P = np.dot(np.dot(F, P), F.T) + Q
        
        # Update
        y = data[i] - np.dot(H, x)
        S = np.dot(np.dot(H, P), H.T) + R
        K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
        x = x + np.dot(K, y)
        P = np.dot((np.eye(2) - np.dot(K, H)), P)
        
        filtered_data.append(x[0][0])
    
    return filtered_data

# Apply Kalman filter to data1 and data2
filtered_data1 = kalman_filter(data1[2])
filtered_data2 = kalman_filter(data2[2])

# Plot the graph before and after filter for radar 0
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(data1[0], data1[2], label='Original Data')
plt.xlabel('Timestamp')
plt.ylabel('Angle')
plt.title('Radar 0 - Before Filter')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(data1[0], filtered_data1, label='Filtered Data')
plt.xlabel('Timestamp')
plt.ylabel('Angle')
plt.title('Radar 0 - After Filter')
plt.legend()

plt.tight_layout()
plt.show()

# Plot the graph before and after filter for radar 1
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(data2[0], data2[2], label='Original Data')
plt.xlabel('Timestamp')
plt.ylabel('Angle')
plt.title('Radar 1 - Before Filter')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(data2[0], filtered_data2, label='Filtered Data')
plt.xlabel('Timestamp')
plt.ylabel('Angle')
plt.title('Radar 1 - After Filter')
plt.legend()

plt.tight_layout()
plt.show()
