#Sensor fusion for Magnetic and Em 30 
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt

def sensor_fusion(Em_data, magnetic_data):
    min_length = min(len(Em_data), len(magnetic_data))

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
        y = np.array([[Em_data[i, 0]], [Em_data[i, 1]]]) - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ linalg.inv(S)
        x = x + K @ y
        P = (np.eye(2) - K @ H) @ P

        w_Em = K[0, 0]  # Weight for vision data
        w_magnetic = K[0, 0]  # Weight for magnetic data

        print("Time Step:", i)
        print("Weight for EM Data:", w_Em)
        print("Weight for Magnetic Data:", w_magnetic)
        
        fused_value = w_Em * Em_data[i] + w_magnetic * magnetic_data[i]
        fused_data.append(fused_value)

    return np.array(fused_data)

Em_data_path = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\EMdata_30deg.csv"  # Replace with the actual path
magnetic_data_path = r"C:\Users\SHAIK VAZEER AHAMED\Downloads\magnetic_data_jm - 30.csv"  

Em_data = pd.read_csv(Em_data_path)[["Pos(Mx)", "Pos(My)"]].values  
magnetic_data = pd.read_csv(magnetic_data_path)[["Pos(Mx)", "Pos(My)"]].values  

fused_data = sensor_fusion(Em_data, magnetic_data)

plt.figure()
# Plotting the original vision and magnetic data
plt.plot(Em_data[:, 0], Em_data[:, 1],"r*-", label='Em Data 30')
plt.plot(magnetic_data[:, 0], magnetic_data[:, 1], label='Magnetic Data')

# Plotting the fused data
plt.plot(fused_data[:, 0], fused_data[:, 1],"b", label='Fused Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sensor Fusion using Kalman Filter')
plt.legend()
plt.show()

