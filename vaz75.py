#Sensor fusion for Pos(X),Pos(Y) 
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt

def sensor_fusion(vision_data, magnetic_data):
    min_length = min(len(vision_data), len(magnetic_data))

    dt = 1.0  # Time step
    F = np.array([[1, dt], [0, 1]]) 
    H = np.array([[1, 0], [0, 1]]) 
    Q = np.eye(2)  
    R = np.eye(2)  

    x = np.zeros((2, 1))  
    P = np.eye(2)  

    fused_data = []  

    for i in range(min_length):
        # Prediction step
        x = F @ x
        P = F @ P @ F.T + Q

        # Update step
        y = np.array([[vision_data[i, 0]], [vision_data[i, 1]]]) - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ linalg.inv(S)
        x = x + K @ y
        P = (np.eye(2) - K @ H) @ P

        w_vision = K[0, 0]  # Weight for vision data
        w_magnetic = K[0, 0]  # Weight for magnetic data

        print("Time Step:", i)
        print("Weight for Vision Data:", w_vision)
        print("Weight for Magnetic Data:", w_magnetic)
        
        fused_value = w_vision * vision_data[i] + w_magnetic * magnetic_data[i]
        fused_data.append(fused_value)

    return np.array(fused_data)

vision_data_path = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\vision_EM_data_1june2023.csv"  # Replace with the actual path
magnetic_data_path = r"C:\Users\SHAIK VAZEER AHAMED\Downloads\magnetic_data.csv"  

vision_data = pd.read_csv(vision_data_path)[["X", "Y"]].values  
magnetic_data = pd.read_csv(magnetic_data_path)[["Pos(Mx)", "Pos(My)"]].values  

fused_data = sensor_fusion(vision_data, magnetic_data)

plt.figure()
# Plotting the original vision and magnetic data
plt.plot(vision_data[:, 0], vision_data[:, 1], label='Vision Data')
plt.plot(magnetic_data[:, 0], magnetic_data[:, 1], label='Magnetic Data')

# Plotting the fused data
plt.plot(fused_data[:, 0], fused_data[:, 1], label='Fused Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sensor Fusion using Kalman Filter')
plt.legend()
plt.show()

