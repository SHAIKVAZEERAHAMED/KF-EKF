import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load radar data
dataset1 = pd.read_csv(r'C:\Users\SHAIK VAZEER AHAMED\Downloads\philos_2019_10_29__16_00_19_target_and_sailboats\philos_2019_10_29__16_00_19_target_and_sailboats\csv\radar_0_segments.csv')
dataset2 = pd.read_csv(r'C:\Users\SHAIK VAZEER AHAMED\Downloads\philos_2019_10_29__16_00_19_target_and_sailboats\philos_2019_10_29__16_00_19_target_and_sailboats\csv\radar_1_segments.csv')


# Extract the relevant columns from each dataset
time_dataset1 = dataset1['time']
angle_dataset1 = dataset1['angle']
intensity_columns1 = [col for col in dataset1.columns if col.startswith('intensity')]
intensity_dataset1 = dataset1[intensity_columns1]

time_dataset2 = dataset2['time']
angle_dataset2 = dataset2['angle']
intensity_columns2 = [col for col in dataset2.columns if col.startswith('intensity')]
intensity_dataset2 = dataset2[intensity_columns2]

# Determine the length of the shorter dataset
min_length = min(len(angle_dataset1), len(angle_dataset2))

# Estimate object trajectory
estimated_angles = []
for i in range(min_length):
    intensity1 = intensity_dataset1.iloc[i].values
    intensity2 = intensity_dataset2.iloc[i].values
    angle1 = angle_dataset1.iloc[i]
    angle2 = angle_dataset2.iloc[i]
    
    # Calculate the sum of intensities for each radar
    intensity_sum1 = np.sum(intensity1)
    intensity_sum2 = np.sum(intensity2)
    
    # Choose the angle based on the radar with higher intensity sum
    if intensity_sum1 > intensity_sum2:
        estimated_angles.append(angle1)
    else:
        estimated_angles.append(angle2)

# Plot the estimated object trajectory and the angles from both radars
plt.figure()
plt.plot(time_dataset1[:min_length], angle_dataset1[:min_length], 'bo-', label='Radar 1 Angle')
plt.plot(time_dataset2[:min_length], angle_dataset2[:min_length], 'ro-', label='Radar 2 Angle')
plt.plot(time_dataset1[:min_length], estimated_angles, 'g-', label='Estimated Trajectory')
plt.xlabel('Time')
plt.ylabel('Angle')
plt.title('Object Trajectory')
plt.legend()
plt.show()
