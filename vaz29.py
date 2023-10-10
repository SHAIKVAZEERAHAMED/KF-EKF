import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# Load radar data
dataset1 = pd.read_csv(r'C:\Users\SHAIK VAZEER AHAMED\Downloads\philos_2019_10_29__16_00_19_target_and_sailboats\philos_2019_10_29__16_00_19_target_and_sailboats\csv\radar_0_segments.csv')
dataset2 = pd.read_csv(r'C:\Users\SHAIK VAZEER AHAMED\Downloads\philos_2019_10_29__16_00_19_target_and_sailboats\philos_2019_10_29__16_00_19_target_and_sailboats\csv\radar_1_segments.csv')

# Extract the relevant columns from each dataset
time_dataset1 = dataset1['time']
angle_dataset1 = dataset1['angle']
intensity_cols1 = [col for col in dataset1.columns if col.startswith('intensity')]
intensity_dataset1 = dataset1[intensity_cols1]

time_dataset2 = dataset2['time']
angle_dataset2 = dataset2['angle']
intensity_cols2 = [col for col in dataset2.columns if col.startswith('intensity')]
intensity_dataset2 = dataset2[intensity_cols2]

# Determine the length of the shorter dataset
min_length = min(len(angle_dataset1), len(angle_dataset2))

# Create a Kalman filter
dim_z = 1  # Angle measurement
kf = KalmanFilter(dim_x=2, dim_z=dim_z)

# Define the state transition matrix
kf.F = np.array([[1, 1],
                 [0, 1]])

# Define the measurement matrix
kf.H = np.array([[1, 0]])

# Define the measurement noise covariance matrix
kf.R = np.array([[0.1]])

# Define the initial state and covariance matrix
kf.x = np.array([0, 0])
kf.P = np.array([[1, 0],
                 [0, 1]])

# Perform the estimation
estimated_angles = []
selected_angles = []

for i in range(min_length):
    intensity1 = intensity_dataset1.iloc[i].values
    intensity2 = intensity_dataset2.iloc[i].values
    
    # Determine the index of the maximum intensity for each radar
    index1 = np.argmax(intensity1)
    index2 = np.argmax(intensity2)
    
    # Select the angle based on the radar with higher intensity
    if intensity1[index1] > intensity2[index2]:
        selected_angle = angle_dataset1.iloc[i]
        z = np.array([[angle_dataset1.iloc[i]]])
    else:
        selected_angle = angle_dataset2.iloc[i]
        z = np.array([[angle_dataset2.iloc[i]]])
    
    # Predict the state
    kf.predict()
    
    # Update the state based on the selected angle measurement
    kf.update(z)
    
    # Get the estimated angle
    estimated_angle = kf.x[0]
    
    selected_angles.append(selected_angle)
    estimated_angles.append(estimated_angle)
    
    # Plot the estimated object trajectory and the selected angle
    plt.plot(time_dataset1[:i+1], selected_angles, 'bo-', label='Selected Angle')
    plt.plot(time_dataset1[:i+1], estimated_angles, 'g-', label='Estimated Trajectory')
    plt.xlabel('Time')
    plt.ylabel('Angle')
    plt.title('Object Trajectory (Kalman Filter)')
    plt.legend()
    plt.pause(0.1)  # Pause for a short duration between each plot
    plt.clf()  # Clear the figure for the next plot
plt.show()
