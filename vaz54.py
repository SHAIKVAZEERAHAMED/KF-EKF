import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt

# File path of the dataset Vision data_zerodes
file_path1 = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\vision_EM_data_1june2023.csv"
# File path of the dataset EMdata _zero deg
file_path2 = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\EMdata _zero deg.csv"

path1_data = pd.read_csv(file_path1)
path2_data = pd.read_csv(file_path2)

sno_path1 = path1_data['sno_v']
angle_measurements_path1 = path1_data['ang_v']

sno_path2 = path2_data['sno_m']
angle_measurements_path2 = path2_data['ang_m']

# Combine the angle measurements from both paths
sno = pd.concat([sno_path1, sno_path2])
angle_measurements = pd.concat([angle_measurements_path1, angle_measurements_path2])

# Define the EKF class
class ExtendedKalmanFilter:
    def __init__(self):
        # Initialize state variables
        self.state = None
        self.covariance = None

    def initialize(self, initial_state, initial_covariance):
        # Set initial state and covariance
        self.state = initial_state
        self.covariance = initial_covariance

    def predict(self, F, Q):
        # Predict the next state based on the system dynamics
        self.state = np.dot(F, self.state)
        self.covariance = np.dot(np.dot(F, self.covariance), F.T) + Q

    def update(self, H, R, z):
        # Update the state based on the sensor measurement
        y = z - np.dot(H, self.state)
        S = np.dot(np.dot(H, self.covariance), H.T) + R
        K = np.dot(np.dot(self.covariance, H.T), np.linalg.inv(S))
        self.state += np.dot(K, y)
        self.covariance = np.dot((np.eye(self.state.shape[0]) - np.dot(K, H)), self.covariance)

# Define measurement models and noise characteristics
H_vision = np.array([[1]])  # Vision measurement model
H_em = np.array([[1]])  # EM measurement model
R_vision = np.array([[0.01]])  # Vision measurement noise covariance
R_em = np.array([[0.1]])  # EM measurement noise covariance

# Initialize EKF
ekf = ExtendedKalmanFilter()

# Define initial state and covariance
initial_state = np.array([[0.0]])  # Initial angle estimate
initial_covariance = np.array([[1]])  # Initial covariance estimate
ekf.initialize(initial_state, initial_covariance)

# Initialize lists to store estimated angles and original angles
estimated_angles_fusion = []
original_angles1 = []
original_angles2 = []

# Perform sensor fusion for each measurement
for i in range(len(path2_data)):
    # Get measurements from vision and EM sensors
    vision_measurement = angle_measurements_path1[i]  # Assuming 'ang_v' column contains vision measurements
    em_measurement = angle_measurements_path2[i]  # Assuming 'ang_m' column contains EM measurements

    # Perform EKF prediction
    F = np.array([[1]])  # State transition matrix (assuming constant velocity)
    Q = np.array([[0.01]])  # Process noise covariance (tune as per system dynamics)
    ekf.predict(F, Q)

    # Perform sensor fusion in the update step
    H_fusion = np.array([[1]])  # Measurement model for fusion
    R_fusion = np.array([[0.01]])  # Measurement noise covariance for fusion
    measurement_fusion = np.array([[vision_measurement], [em_measurement]])  # Combined measurement

    H_combined = np.array([[1], [1]])  # Combine both H matrices
    R_combined = np.array([[0.01, 0], [0, 0.1]])  # Combine both R matrices

    ekf.update(H_combined, R_combined, measurement_fusion)
    estimated_angles_fusion.append(ekf.state[0, 0])

    # Store original angles
    original_angles1.append(angle_measurements_path1[i])  # Assuming 'ang_v' column contains original angles
    original_angles2.append(angle_measurements_path2[i])

# Plot estimated angles from sensor fusion and original angles
plt.plot(estimated_angles_fusion, label='Estimated Angle (Fusion)')
plt.plot(original_angles1, label='Original Angle of Vision')
plt.plot(original_angles2, label='Original Angle of EM')
plt.xlabel('Sno')
plt.ylabel('Angle')
plt.legend()
plt.show()
