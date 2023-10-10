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
angle_measurements_path1 = path1_data[['X', 'Y']]

sno_path2 = path2_data['sno_m']
angle_measurements_path2 = path2_data[['Pos(Mx)', 'Pos(My)']]

sno = pd.concat([sno_path1, sno_path2])
angle_measurements = pd.concat([angle_measurements_path1, angle_measurements_path2])

class ExtendedKalmanFilter:
    def __init__(self):
        self.state = None
        self.covariance = None

    def initialize(self, initial_state, initial_covariance):
        self.state = initial_state
        self.covariance = initial_covariance

    def predict(self, F, Q):
        self.state = np.dot(F, self.state)
        self.covariance = np.dot(np.dot(F, self.covariance), F.T) + Q

    def update(self, H, R, z):
        y = z - np.dot(H, self.state)
        S = np.dot(np.dot(H, self.covariance), H.T) + R
        K = np.dot(np.dot(self.covariance, H.T), np.linalg.inv(S))
        self.state += np.dot(K, y)
        self.covariance = np.dot((np.eye(self.state.shape[0]) - np.dot(K, H)), self.covariance)

H_vision = np.array([[1, 0], [0, 1]])
H_em = np.array([[1, 0], [0, 1]])
R_vision = np.array([[0.01], [0.01]])
R_em = np.array([[0.1], [0.1]])

ekf = ExtendedKalmanFilter()

initial_state = np.array([[0.0], [0.0]])
initial_covariance = np.array([[1, 0], [0, 1]])
ekf.initialize(initial_state, initial_covariance)

estimated_positions_fusion = []
original_positions1 = []
original_positions2 = []

for i in range(len(path1_data)):
    vision_measurement = angle_measurements_path1.iloc[i]
    em_measurement = angle_measurements_path2.iloc[i]

    F = np.array([[1, 0], [0, 1]])
    Q = np.array([[0.01, 0], [0, 0.01]])
    ekf.predict(F, Q)

    H_fusion = np.array([[1, 0], [0, 1]])
    R_fusion = np.array([[0.01, 0], [0, 0.01]])
    measurement_fusion = np.array([[vision_measurement[0]], [em_measurement[0]]])

    H_combined = np.array([[1, 0], [0, 1]])
    R_combined = np.array([[0.01, 0], [0, 0.1]])

    ekf.update(H_combined, R_combined, measurement_fusion)
    estimated_positions_fusion.append(ekf.state[:, 0])

    original_positions1.append(angle_measurements_path1.iloc[i])
    original_positions2.append(angle_measurements_path2.iloc[i])

estimated_positions_fusion = np.array(estimated_positions_fusion)
original_positions1 = np.array(original_positions1)
original_positions2 = np.array(original_positions2)
plt.figure()
plt.plot(estimated_positions_fusion[:, 0], estimated_positions_fusion[:, 1], label='Estimated Positions (Fusion)')
plt.plot(original_positions1[:, 0], original_positions1[:, 1], label='Original Positions of Vision')
plt.plot(original_positions2[:, 0], original_positions2[:, 1], label='Original Positions of EM')
plt.xlabel('Pos(Mx)')
plt.ylabel('Pos(My)')
plt.legend()

errors_vision = estimated_positions_fusion - original_positions1
errors_em = estimated_positions_fusion - original_positions2

print("Sensor Fusion Results:")
for i in range(len(estimated_positions_fusion)):
    print(f"Step {i+1}:")
    print("Estimate - Original Position Error (Vision):", errors_vision[i])
    print("Estimate - Original Position Error (EM):", errors_em[i])
    print()

path_data = pd.read_csv(file_path2)

sno = path_data['sno_m']
angle_measurements = path_data[['Pos(Mx)', 'Pos(My)']]

class ExtendedKalmanFilter:
    def __init__(self):
        self.state = None
        self.covariance = None

    def initialize(self, initial_state, initial_covariance):
        self.state = initial_state
        self.covariance = initial_covariance

    def predict(self, F, Q):
        self.state = np.dot(F, self.state)
        self.covariance = np.dot(np.dot(F, self.covariance), F.T) + Q

    def update(self, H, R, z):
        y = z - np.dot(H, self.state)
        S = np.dot(np.dot(H, self.covariance), H.T) + R
        K = np.dot(np.dot(self.covariance, H.T), np.linalg.inv(S))
        self.state += np.dot(K, y)
        self.covariance = np.dot((np.eye(self.state.shape[0]) - np.dot(K, H)), self.covariance)

H_em = np.array([[1, 0], [0, 1]])
R_em = np.array([[0.1], [0.1]])

ekf = ExtendedKalmanFilter()

initial_state = np.array([[0.0], [0.0]])
initial_covariance = np.array([[1, 0], [0, 1]])
ekf.initialize(initial_state, initial_covariance)

estimated_positions_em = []
original_positions_em = []

for i in range(len(path_data)):
    em_measurement = angle_measurements.iloc[i]

    F = np.array([[1, 0], [0, 1]])
    Q = np.array([[0.01, 0], [0, 0.01]])
    ekf.predict(F, Q)

    H_em = np.array([[1, 0], [0, 1]])
    R_em = np.array([[0.1, 0], [0, 0.1]])
    measurement_em = np.array([[em_measurement[0]], [em_measurement[1]]])

    ekf.update(H_em, R_em, measurement_em)
    estimated_positions_em.append(ekf.state[:, 0])

    original_positions_em.append(em_measurement)

estimated_positions_em = np.array(estimated_positions_em)
original_positions_em = np.array(original_positions_em)

plt.figure()
plt.plot(estimated_positions_em[:, 0], estimated_positions_em[:, 1], label='Estimated Positions (EM)')
plt.plot(original_positions_em[:, 0], original_positions_em[:, 1], label='Original Positions of EM')
plt.xlabel('Pos(Mx)')
plt.ylabel('Pos(My)')
plt.legend()

path_data = pd.read_csv(file_path1)

sno = path_data['sno_v']
angle_measurements = path_data[['X', 'Y']]

class ExtendedKalmanFilter:
    def __init__(self):
        self.state = None
        self.covariance = None

    def initialize(self, initial_state, initial_covariance):
        self.state = initial_state
        self.covariance = initial_covariance

    def predict(self, F, Q):
        self.state = np.dot(F, self.state)
        self.covariance = np.dot(np.dot(F, self.covariance), F.T) + Q

    def update(self, H, R, z):
        y = z - np.dot(H, self.state)
        S = np.dot(np.dot(H, self.covariance), H.T) + R
        K = np.dot(np.dot(self.covariance, H.T), np.linalg.inv(S))
        self.state += np.dot(K, y)
        self.covariance = np.dot((np.eye(self.state.shape[0]) - np.dot(K, H)), self.covariance)

H_em = np.array([[1, 0], [0, 1]])
R_em = np.array([[0.1], [0.1]])

ekf = ExtendedKalmanFilter()

initial_state = np.array([[0.0], [0.0]])
initial_covariance = np.array([[1, 0], [0, 1]])
ekf.initialize(initial_state, initial_covariance)

estimated_positions_em = []
original_positions_em = []

for i in range(len(path_data)):
    em_measurement = angle_measurements.iloc[i]

    F = np.array([[1, 0], [0, 1]])
    Q = np.array([[0.01, 0], [0, 0.01]])
    ekf.predict(F, Q)

    H_em = np.array([[1, 0], [0, 1]])
    R_em = np.array([[0.1, 0], [0, 0.1]])
    measurement_em = np.array([[em_measurement[0]], [em_measurement[1]]])

    ekf.update(H_em, R_em, measurement_em)
    estimated_positions_em.append(ekf.state[:, 0])

    original_positions_em.append(em_measurement)

estimated_positions_em = np.array(estimated_positions_em)
original_positions_em = np.array(original_positions_em)

plt.figure()
plt.plot(estimated_positions_em[:, 0], estimated_positions_em[:, 1], label='Estimated Positions Vision')
plt.plot(original_positions_em[:, 0], original_positions_em[:, 1], label='Original Positions of Vision')
plt.xlabel('Pos(Mx)')
plt.ylabel('Pos(My)')
plt.legend()
plt.show()
