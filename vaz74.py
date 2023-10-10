import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

def sensor_fusion(vision_data, magnetic_data):
    # Determine the minimum length
    min_length = min(len(vision_data), len(magnetic_data))

    fused_data = np.zeros_like(vision_data)  # Array to store fused data

    # Perform sensor fusion for each time step up to the minimum length
    for i in range(min_length):
        # Calculate Euclidean distance between vision and magnetic data
        distance = euclidean(vision_data[i], magnetic_data[i])
        print(distance)
        # Calculate weights based on the inverse of the distance
        w_vision = 1 / (1 + distance)  # Weight for vision data
        w_magnetic = 1 - w_vision  # Weight for magnetic data

        # Print the weights for the current time step
        print("Time Step:", i)
        print("Weight for Vision Data:", w_vision)
        print("Weight for Magnetic Data:", w_magnetic)

        # Calculate weighted average using the fused data
        fused_value = w_vision * vision_data[i] + w_magnetic * magnetic_data[i]
        fused_data[i] = fused_value

    return fused_data

# Load vision and magnetic data from CSV files
vision_data_path = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\vision_EM_data_1june2023.csv"  # Replace with the actual path
magnetic_data_path = r"C:\Users\SHAIK VAZEER AHAMED\Downloads\magnetic_data.csv"  # Replace with the actual path

vision_data = pd.read_csv(vision_data_path)[["X", "Y"]].values  # Assuming "X" and "Y" columns contain the vision data
magnetic_data = pd.read_csv(magnetic_data_path)[["Pos(Mx)", "Pos(My)"]].values  # Assuming "Pos(X)" and "Pos(Y)" columns contain the magnetic data

# Perform sensor fusion using Euclidean distance
fused_data = sensor_fusion(vision_data, magnetic_data)

plt.figure()
# Plotting the original vision and magnetic data
plt.plot(vision_data[:, 0], vision_data[:, 1], label='Vision Data')
plt.plot(magnetic_data[:, 0], magnetic_data[:, 1], label='Magnetic Data')

# Plotting the fused data
plt.plot(fused_data[:, 0], fused_data[:, 1], label='Fused Data')
plt.xlabel('Pos(X)')
plt.ylabel('Pos(Y)')
plt.title('Sensor Fusion using Euclidean Distance')
plt.legend()
plt.show()
