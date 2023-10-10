# KF Magnetic data 0
import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt

# File path of the dataset
file_path1 = r"C:\Users\SHAIK VAZEER AHAMED\Downloads\magnetic_data.csv"

# Load the data from CSV file
data1 = pd.read_csv(file_path1)

# Create Kalman filter
num_states1 = 5  # Number of states to track (Pos(Mx), Pos(My), Pos(Mz), dis_m, ang_m)
dim_z1 = 5  # Number of measurements (Pos(Mx), Pos(My), Pos(Mz), dis_m, ang_m)

kf1 = KalmanFilter(dim_x=num_states1, dim_z=dim_z1)

# Initialize Kalman filter
initial_state1 = np.array([data1['Pos(Mx)'].iloc[0], data1['Pos(My)'].iloc[0], data1['Pos(Mz)'].iloc[0], data1['dis_m'].iloc[0], data1['ang_m'].iloc[0]])
initial_covariance1 = np.eye(num_states1) * 0.1  # Adjust the initial covariance as needed

kf1.x = initial_state1
kf1.P = initial_covariance1

# Define the state transition matrix
dt = 1  # Time step
kf1.F = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

# Define the measurement matrix
kf1.H = np.eye(dim_z1)

# Define the process noise covariance matrix
process_noise1 = np.eye(num_states1) * 0.01  # Adjust the process noise covariance as needed
kf1.Q = process_noise1

# Define the measurement noise covariance matrix
measurement_noise1 = np.eye(dim_z1) * 0.1  # Adjust the measurement noise covariance as needed
kf1.R = measurement_noise1

# Apply Kalman filter to estimate the true values
estimated_values1 = []
for _, row in data1.iterrows():
    measurement1 = np.array([
    row['Pos(Mx)'],
    row['Pos(My)'],
    row['Pos(Mz)'],
    row['dis_m'],
    row['ang_m']
    ])

    # Predict the object's state
    kf1.predict()

    # Update the object's state based on the measurement
    kf1.update(measurement1)

    # Save the estimated state
    estimated_values1.append(kf1.x.copy())


estimated_values1 = np.array(estimated_values1)

# Plot the original and estimated values
plt.figure()
plt.plot(data1['Pos(Mx)'], data1['Pos(My)'], 'g*-', label='Original')
plt.plot(estimated_values1[:, 0], estimated_values1[:, 1], 'y*-', label='Estimated')

plt.xlabel('Position (Mx)')
plt.ylabel('Position (My)')
plt.title('Original and Estimated Positions')
plt.legend()

plt.figure()
plt.plot(data1['sno_m'], data1['dis_m'], 'yo-', label='Original dis_m')
plt.plot(data1['sno_m'], estimated_values1[:, 3], 'r*-', label='Estimated dis_m')

plt.xlabel('Sample Number')
plt.ylabel('dis_m ')
plt.title('Original and Estimated Distance')
plt.legend()

plt.figure()
plt.plot(data1['sno_m'], data1['ang_m'], 'mo-', label='Original ang_m')
plt.plot(data1['sno_m'], estimated_values1[:, 4], 'r*-', label='Estimated ang_m')
plt.xlabel('Sample Number')
plt.ylabel('ang_m')
plt.title('Original and Estimated Angle')
plt.legend()
plt.show()
