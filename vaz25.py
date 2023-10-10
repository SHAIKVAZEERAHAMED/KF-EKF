import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV files
csv_file1 = r'C:\Users\SHAIK VAZEER AHAMED\Downloads\philos_2019_10_29__16_00_19_target_and_sailboats\philos_2019_10_29__16_00_19_target_and_sailboats\csv\radar_0_segments.csv'
csv_file2 = r'C:\Users\SHAIK VAZEER AHAMED\Downloads\philos_2019_10_29__16_00_19_target_and_sailboats\philos_2019_10_29__16_00_19_target_and_sailboats\csv\radar_1_segments.csv'

data1 = pd.read_csv(csv_file1)
data2 = pd.read_csv(csv_file2)

# Convert specific columns to numeric format
data1['angle'] = pd.to_numeric(data1['angle'], errors='coerce')
data2['angle'] = pd.to_numeric(data2['angle'], errors='coerce')

# Apply Kalman filter
def kalman_filter(data):
    n = len(data)
    
    # Initialize Kalman filter parameters
    # State vector [position, velocity]
    x = np.array([[data.iloc[0, 0]], [0]], dtype=float)
    
    # State transition matrix
    F = np.array([[1, 1], [0, 1]], dtype=float)
    
    # Observation matrix
    H = np.array([[1, 0]], dtype=float)
    
    # Measurement noise covariance matrix
    R = np.array([[1]], dtype=float)
    
    # Process noise covariance matrix
    Q = np.array([[0.001, 0], [0, 0.001]], dtype=float)
    
    # Initial state covariance matrix
    P = np.eye(2)
    
    filtered_data = []
    
    for i in range(n):
        # Predict
        x = np.dot(F, x)
        P = np.dot(np.dot(F, P), F.T) + Q
        
        # Update
        y = data.iloc[i, 1] - np.dot(H, x)
        S = np.dot(np.dot(H, P), H.T) + R
        K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
        x = x + np.dot(K, y)
        P = np.dot((np.eye(2) - np.dot(K, H)), P)
        
        filtered_data.append(x[0][0])
    
    return filtered_data

# Apply Kalman filter to data1 and data2
filtered_data1 = kalman_filter(data1)
filtered_data2 = kalman_filter(data2)

# Plot the graph before and after filter for radar 0 (angle and intensity)
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(data1['time'], data1['angle'], label='Original Angle')
plt.xlabel('Timestamp')
plt.ylabel('Angle')
plt.title('Radar 0 - Before Filter (Angle)')
plt.legend()

for i in range(512):
    plt.subplot(2, 512, 512+i+1)
    plt.plot(data1['time'], data1[f'intensity{i}'], label=f'Original Intensity {i}')
    plt.xlabel('Timestamp')
    plt.ylabel('Intensity')
    plt.title(f'Radar 0 - Before Filter (Intensity {i})')
    plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(data1['time'], filtered_data1, label='Filtered Angle')
plt.xlabel('Timestamp')
plt.ylabel('Angle')
plt.title('Radar 0 - After Filter (Angle)')
plt.legend()

for i in range(512):
    plt.subplot(2, 512, 512+i+1)
    plt.plot(data1['time'], data1[f'intensity{i}'], label=f'Original Intensity {i}')
    plt.xlabel('Timestamp')
    plt.ylabel('Intensity')
    plt.title(f'Radar 0 - Before Filter (Intensity {i})')
    plt.legend()

plt.tight_layout()
plt.show()

# Plot the graph before and after filter for radar 1 (angle and intensity)
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(data2['time'], data2['angle'], label='Original Angle')
plt.xlabel('Timestamp')
plt.ylabel('Angle')
plt.title('Radar 1 - Before Filter (Angle)')
plt.legend()

for i in range(512):
    plt.subplot(2, 512, 512+i+1)
    plt.plot(data2['time'], data2[f'intensity{i}'], label=f'Original Intensity {i}')
    plt.xlabel('Timestamp')
    plt.ylabel('Intensity')
    plt.title(f'Radar 1 - Before Filter (Intensity {i})')
    plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(data2['time'], filtered_data2, label='Filtered Angle')
plt.xlabel('Timestamp')
plt.ylabel('Angle')
plt.title('Radar 1 - After Filter (Angle)')
plt.legend()

for i in range(512):
    plt.subplot(2, 512, 512+i+1)
    plt.plot(data2['time'], data2[f'intensity{i}'], label=f'Original Intensity {i}')
    plt.xlabel('Timestamp')
    plt.ylabel('Intensity')
    plt.title(f'Radar 1 - Before Filter (Intensity {i})')
    plt.legend()

plt.tight_layout()
plt.show()
