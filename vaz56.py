import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt

# File path of the dataset EMdata _zero deg
file_path = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\EMdata _zero deg.csv"

path_data = pd.read_csv(file_path)

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

plt.plot(estimated_positions_em[:, 0], estimated_positions_em[:, 1], label='Estimated Positions (EM)')
plt.plot(original_positions_em[:, 0], original_positions_em[:, 1], label='Original Positions of EM')
plt.xlabel('Pos(Mx)')
plt.ylabel('Pos(My)')
plt.legend()

plt.show()
