# Sensor fusion assumed magnetic weight is more
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt

def sensor_fusion(vision_data, magnetic_data):
    # Determine the minimum length
    min_length = min(len(vision_data), len(magnetic_data))

    # Initialize Kalman filter parameters
    dt = 1.0  # Time step
    F = np.array([[1, dt], [0, 1]])  # State transition matrix
    H = np.array([[1, 0]])  # Measurement matrix
    Q = np.eye(2)  # Process noise covariance
    R = np.eye(1)  # Measurement noise covariance

    # Initialize state and covariance matrices
    x = np.zeros((2, 1))  # Initial state vector
    P = np.eye(2)  # Initial covariance matrix

    fused_data = []  # List to store fused data

    # Perform sensor fusion for each time step up to the minimum length
    for i in range(min_length):
        # Prediction step
        x = F @ x
        P = F @ P @ F.T + Q

        # Update step
        y = vision_data[i] - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ linalg.inv(S)
        x = x + K @ y
        P = (np.eye(2) - K @ H) @ P

        # Calculate weights based on the Kalman gain
        w_vision = K[0, 0]  # Weight for vision data
        w_magnetic = K[0, 0]  # Weight for magnetic data
        # Print the weights for the current time step
        print("Time Step:", i)
        print("Weight for Vision Data:", w_vision)
        print("Weight for Magnetic Data:", w_magnetic)
        # Calculate weighted average using the fused data
        fused_value = w_vision * vision_data[i] + w_magnetic * magnetic_data[i]
        fused_data.append(fused_value)

    return fused_data


# Load vision and magnetic data from CSV files
vision_data_path = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\vision_EM_data_1june2023.csv"  # Replace with the actual path
magnetic_data_path = r"C:\Users\SHAIK VAZEER AHAMED\Downloads\magnetic_data.csv"  # Replace with the actual path

vision_data1 = pd.read_csv(vision_data_path)["dis_v"].values  # Assuming "X" column contains the vision data
magnetic_data1 = pd.read_csv(magnetic_data_path)["dis_m"].values  # Assuming "Pos(Mx)" column contains the magnetic data

# Perform sensor fusion using Kalman filter
fused_data1 = sensor_fusion(vision_data1, magnetic_data1)

plt.figure()
# Plotting the original vision and magnetic data
plt.plot(vision_data1, label='Vision Data')
plt.plot(magnetic_data1, label='Sonar Data')

# Plotting the fused data
plt.plot(fused_data1, label='Fused Data')
plt.xlabel('Time Step')
plt.ylabel('Distance')
plt.title('Sensor Fusion using Kalman Filter (distance)')
plt.legend()
plt.show()
