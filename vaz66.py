import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter
import matplotlib.pyplot as plt

file_path = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\Battery.csv"

data = pd.read_csv(file_path)
data['Battery'] = data['Battery'].astype(float)

num_states = 2  
dim_z = 1  

kf = KalmanFilter(dim_x=num_states, dim_z=dim_z)

initial_state = np.array([data['Battery'].iloc[0], 0.0])
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
    measurement = np.array([row['Battery']])

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

original_mean = np.mean(data['Battery'])
original_variance = np.var(data['Battery'])
original_std = np.std(data['Battery'])

kf_mean = np.mean(kf_estimated_values[:, 0])
kf_variance = np.var(kf_estimated_values[:, 0]) 
kf_std = np.std(kf_estimated_values[:, 0])
kf_covariance = np.cov(data['Battery'], kf_estimated_values[:, 0])[0, 1] 

ekf_mean = np.mean(ekf_estimated_values[:, 0])
ekf_variance = np.var(ekf_estimated_values[:, 0]) 
ekf_std = np.std(ekf_estimated_values[:, 0]) 
ekf_covariance = np.cov(data['Battery'], ekf_estimated_values[:, 0])[0, 1] 

print("Original Data:")
print("Mean:", original_mean)
print("Variance:", original_variance)
print("Standard Deviation:", original_std)
print()

print("KF Estimated Data:")
print("Mean:", kf_mean)
print("Variance:", kf_variance)
print("Standard Deviation:", kf_std)
print("Covariance with Original Data:", kf_covariance)
print()

print("EKF Estimated Data:")
print("Mean:", ekf_mean)
print("Variance:", ekf_variance)
print("Standard Deviation:", ekf_std)
print("Covariance with Original Data:", ekf_covariance)

plt.plot(data['Battery'], 'g*-', label='Original')
plt.plot(kf_estimated_values[:, 0], 'r*-', label='KF Estimated')
plt.plot(ekf_estimated_values[:, 0], 'bo-', label='EKF Estimated')

plt.xlabel('Samples')
plt.ylabel('Battery (V)')
plt.title('Original, KF-Estimated, and EKF-Estimated Multimeter Values')
plt.legend()
plt.show()
