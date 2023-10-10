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

        # Calculate the fused value using the updated state estimate
        fused_value = float(H @ x)  # Convert to scalar value
        fused_data.append(fused_value)

    return fused_data

vision_data_path = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\vision_EM_data_1june2023.csv"
magnetic_data_path = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\EMdata _zero deg.csv"

vision_data = pd.read_csv(vision_data_path)["dis_v"].values  # Assuming "X" column contains the vision data
magnetic_data = pd.read_csv(magnetic_data_path)["dis_m"].values  # Assuming "Pos(Mx)" column contains the magnetic data

# Perform sensor fusion using Kalman filter
fused_data = sensor_fusion(vision_data, magnetic_data)

# Plotting the original vision and magnetic data
plt.plot(vision_data, label='Vision Data')
plt.plot(magnetic_data, label='Magnetic Data')

# Plotting the fused data
plt.plot(fused_data, label='Fused Data')
plt.xlabel('Time Step')
plt.ylabel('Data')
plt.title('Sensor Fusion using Kalman Filter')
plt.legend()
plt.show()
