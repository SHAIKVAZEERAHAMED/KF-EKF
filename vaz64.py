import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter
import matplotlib.pyplot as plt

file_path = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\Multimeter.csv"

data = pd.read_csv(file_path)
data['Multimeter'] = data['Multimeter'].astype(float)

num_states = 2  
dim_z = 1  

kf = KalmanFilter(dim_x=num_states, dim_z=dim_z)

initial_state = np.array([data['Multimeter'].iloc[0], 0.0])
initial_covariance = np.eye(num_states) * 0.1  

kf.x = initial_state
kf.P = initial_covariance

dt = 1  
kf.F = np.array([[1, dt],
                 [0, 1]])

kf.H = np.array([[1, 0]])

process_noise = np.eye(num_states) * 0.01  
kf.Q = process_noise

measurement_noise = np.eye(dim_z) * 0.1  
kf.R = measurement_noise

ekf = ExtendedKalmanFilter(dim_x=num_states, dim_z=dim_z)

ekf.x = initial_state
ekf.P = initial_covariance

def state_transition_function(x, dt):
    F = np.array([[1, dt],
                  [0, 1]])
    return np.dot(F, x)

def measurement_function(x):
    return np.array([x[0]])

def measurement_jacobian(x):
    return np.array([[1, 0]])

ekf.f = state_transition_function
ekf.h = measurement_function
ekf.H = measurement_jacobian

ekf.Q = process_noise
ekf.R = measurement_noise

kf_estimated_values = []
ekf_estimated_values = []
for _, row in data.iterrows():
    measurement = np.array([row['Multimeter']])

    # Predict and update using Kalman Filter
    kf.predict()
    kf.update(measurement)
    kf_estimated_values.append(kf.x.copy())

    # Predict and update using Extended Kalman Filter
    ekf.predict()
    ekf.update(measurement,measurement_jacobian,measurement_function)
    ekf_estimated_values.append(ekf.x.copy())

kf_estimated_values = np.array(kf_estimated_values)
ekf_estimated_values = np.array(ekf_estimated_values)

plt.plot(data['Multimeter'], 'g*-', label='Original')
plt.plot(kf_estimated_values[:, 0], 'r*-', label='KF Estimated')
plt.plot(ekf_estimated_values[:, 0], 'bo-', label='EKF Estimated')

plt.xlabel('Samples')
plt.ylabel('Multimeter')
plt.title('Original, KF-Estimated, and EKF-Estimated Multimeter Values')
plt.legend()
plt.show()
