import pandas as pd
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
import matplotlib.pyplot as plt

file_path = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\vision_EM_data_1june2023.csv"

data = pd.read_csv(file_path)

num_states = 4
dim_z = 4

ekf = ExtendedKalmanFilter(dim_x=num_states, dim_z=dim_z)

initial_state = np.array([data['X'].iloc[0], data['Y'].iloc[0], data['dis_v'].iloc[0], data['ang_v'].iloc[0]])
initial_covariance = np.eye(num_states) * 0.1

ekf.x = initial_state
ekf.P = initial_covariance

def state_transition_function(x, dt):
    x_next = x.copy()
    x_next[0] += x[2] * np.cos(x[3]) * dt
    x_next[1] += x[2] * np.sin(x[3]) * dt
    return x_next

dt = 1

def predicted_measurement(x):
    return x

ekf.f = state_transition_function
ekf.Hx = predicted_measurement

process_noise = np.eye(num_states) * 0.01
ekf.Q = process_noise

measurement_noise = np.eye(dim_z) * 0.1
ekf.R = measurement_noise

def measurement_function(x):
    return np.array([x[0], x[1], x[2], x[3]])

def measurement_jacobian(x):
    return np.array([[1, 0, dt * np.cos(x[3]), -x[2] * np.sin(x[3])],
                     [0, 1, dt * np.sin(x[3]), x[2] * np.cos(x[3])],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0]])

ekf.h = measurement_function
ekf.H = measurement_jacobian

estimated_values = []
for _, row in data.iterrows():
    measurement = np.array([
        row['X'],
        row['Y'],
        row['dis_v'],
        row['ang_v']
    ])

    ekf.predict()
    ekf.update(measurement,measurement_jacobian,measurement_function)

    estimated_values.append(ekf.x.copy())

estimated_values = np.array(estimated_values)

plt.figure()
plt.plot(data['X'], data['Y'], 'g*-', label='Original')
plt.plot(estimated_values[:, 0], estimated_values[:, 1], 'y*-', label='Estimated')

plt.xlabel('Position X')
plt.ylabel('Position Y')
plt.title('Original and Estimated Positions')
plt.legend()

plt.figure()
plt.plot(data['sno_v'], data['dis_v'], 'yo-', label='Original dis_m')
plt.plot(data['sno_v'], estimated_values[:, 2], 'r*-', label='Estimated dis_m')

plt.xlabel('Sample Number')
plt.ylabel('dis_m')
plt.title('Original and Estimated Distance')
plt.legend()

plt.figure()
plt.plot(data['sno_v'], data['ang_v'], 'mo-', label='Original ang_m')
plt.plot(data['sno_v'], estimated_values[:, 3], 'r*-', label='Estimated ang_m')
plt.xlabel('Sample Number')
plt.ylabel('ang_m')
plt.title('Original and Estimated Angle')
plt.legend()
plt.show()

