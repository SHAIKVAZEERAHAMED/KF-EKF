# Vision data_zerodeg
import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt

# File path of the dataset
file_path = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\vision_EM_data_1june2023.csv"

# Load the data from CSV file
data = pd.read_csv(file_path)

# Create Kalman filter
num_states = 4  # Number of states to track (Pos(Mx), Pos(My), ang_m, ang_m)
dim_z = 4  # Number of measurements (Pos(Mx), Pos(My), ang_m, ang_m)

kf = KalmanFilter(dim_x=num_states, dim_z=dim_z)

# Initialize Kalman filter
initial_state = np.array([data['X'].iloc[0], data['Y'].iloc[0], data['dis_v'].iloc[0], data['ang_v'].iloc[0]])
initial_covariance = np.eye(num_states) * 0.1  # Adjust the initial covariance as needed

kf.x = initial_state
kf.P = initial_covariance

# Define the state transition matrix
dt = 1  # Time step
kf.F = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

# Define the measurement matrix
kf.H = np.eye(dim_z)

# Define the process noise covariance matrix
process_noise = np.eye(num_states) * 0.01  # Adjust the process noise covariance as needed
kf.Q = process_noise

# Define the measurement noise covariance matrix
measurement_noise = np.eye(dim_z) * 0.1  # Adjust the measurement noise covariance as needed
kf.R = measurement_noise

# Apply Kalman filter to estimate the true values
estimated_values = []
for _, row in data.iterrows():
    measurement = np.array([
        row['X'],
        row['Y'],
        row['dis_v'],
        row['ang_v']
    ])

    # Predict the object's state
    kf.predict()

    # Update the object's state based on the measurement
    kf.update(measurement)

    # Save the estimated state
    estimated_values.append(kf.x.copy())

estimated_values = np.array(estimated_values)

# Plot the original and estimated values
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
plt.ylabel('dis_m ')
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

# for i in range(len(data['X'])):
#     print(estimated_values[:, 0] - data['X'][i])
#     print(estimated_values[:, 1] - data['Y'][i])

avg_x = sum(estimated_values[:, 0] - data['X']) / len(data['X'])
avg_y = sum(estimated_values[:, 1] - data['Y']) / len(data['Y'])

print("Average difference in X:", avg_x)
print("Average difference in Y:", avg_y)