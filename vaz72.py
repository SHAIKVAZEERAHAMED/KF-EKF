# Sensor fusion assumed vision weight is more
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt


def sensor_fusion(vision_data, magnetic_data):
    dt = 1.0  
    F = np.array([[1, dt], [0, 1]])  
    H = np.array([[1, 0]]) 
    Q = np.eye(2)  
    R = np.eye(1) 


    x = np.zeros((2, 1))  
    P = np.eye(2)  


    fused_data = []  


    for i in range(len(vision_data)):
        # Prediction step
        x = F @ x
        P = F @ P @ F.T + Q


        # Update step
        y = vision_data[i] - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ linalg.inv(S)
        x = x + K @ y
        P = (np.eye(2) - K @ H) @ P


        # Calculate weights based on the Kalman gain
        w_vision = K[0, 0]  # Weight for vision data
        w_magnetic = K[1, 0]  # Weight for magnetic data


        # Calculate weighted average using the fused data
        fused_value = w_vision * vision_data[i] + w_magnetic * magnetic_data[i]
        fused_data.append(fused_value)


    return fused_data
vision_data_path = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\vision_EM_data_1june2023.csv"  
magnetic_data_path = r"C:\Users\SHAIK VAZEER AHAMED\Downloads\magnetic_data.csv" 


vision_data = pd.read_csv(vision_data_path)["X"].values
magnetic_data = pd.read_csv(magnetic_data_path)["Pos(Mx)"].values  


vision_data1 = pd.read_csv(vision_data_path)["dis_v"].values  
magnetic_data1 = pd.read_csv(magnetic_data_path)["dis_m"].values  


# Perform sensor fusion using Kalman filter
fused_data = sensor_fusion(vision_data, magnetic_data)
fused_data1 = sensor_fusion(vision_data1, magnetic_data1)


plt.figure()
# Plotting the original vision and magnetic data
plt.plot(vision_data, label='Vision Data')
plt.plot(magnetic_data, label='Magnetic Data')


# Plotting the fused data
plt.plot(fused_data, label='Fused Data')
plt.xlabel('Time Step')
plt.ylabel('Fused Value')
plt.title('Sensor Fusion using Kalman Filter X')
plt.legend()
plt.show()


plt.figure()
# Plotting the original vision and magnetic data
plt.plot(vision_data1, label='Vision Data')
plt.plot(magnetic_data1, label='Magnetic Data')


# Plotting the fused data
plt.plot(fused_data1, label='Fused Data')
plt.xlabel('Time Step')
plt.ylabel('Fused Value')
plt.title('Sensor Fusion using Kalman Filter (distance)')
plt.legend()
plt.show()