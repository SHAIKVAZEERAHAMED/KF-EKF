# S/N M64
import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter
import matplotlib.pyplot as plt

# File path of the dataset
file_path1 = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\M64_1_S_N.csv"
file_path2 = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\M64_2_S_N.csv"
file_path3 = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\M64_3_S_N.csv"
file_path4 = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\M64_4_S_N.csv"
file_path5 = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\M64_5_S_N.csv"
file_path6 = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\M64_6_S_N.csv"
file_path7 = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\M64_7_S_N.csv"


# Load the data from CSV file
data1 = pd.read_csv(file_path1)
data1['M64_1_S_N'] = data1['M64_1_S_N'].astype(float)
data2 = pd.read_csv(file_path2)
data2['M64_2_S_N'] = data2['M64_2_S_N'].astype(float)
data3 = pd.read_csv(file_path3)
data3['M64_3_S_N'] = data3['M64_3_S_N'].astype(float)
data4 = pd.read_csv(file_path4)
data4['M64_4_S_N'] = data4['M64_4_S_N'].astype(float)
data5 = pd.read_csv(file_path5)
data5['M64_5_S_N'] = data5['M64_5_S_N'].astype(float)
data6 = pd.read_csv(file_path6)
data6['M64_6_S_N'] = data6['M64_6_S_N'].astype(float)
data7 = pd.read_csv(file_path7)
data7['M64_7_S_N'] = data7['M64_7_S_N'].astype(float)

plt.figure()
# Plot the original, KF-estimated, and EKF-estimated values
plt.plot(data1['M64_1_S_N'], label='1 channel')
plt.plot(data2['M64_2_S_N'], label='2 channel')
plt.plot(data3['M64_3_S_N'], label='3 channel')
plt.plot(data4['M64_4_S_N'], label='4 channel')
plt.plot(data5['M64_5_S_N'], label='5 channel')
plt.plot(data6['M64_6_S_N'], label='6 channel')
plt.plot(data7['M64_7_S_N'], label='7 channel')

    
plt.xlabel('Range')
plt.ylabel('Signal to noise ratio (S/N)')
plt.title('M64 channels')
plt.legend()
plt.show()
