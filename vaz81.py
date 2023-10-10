import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt

def sensor_fusion(vision_data, magnetic_data):
    min_length = min(len(vision_data), len(magnetic_data))

    dt = 1.0  # Time step

    # State transition matrix F
    def state_transition_function(x):
        return np.array([[1, dt, 0],
                         [0, 1, dt],
                         [0, 0, 1]]) @ x

    # Measurement function H
    def measurement_function(x):
        return np.array([[x[0, 0]],
                         [x[1, 0]]])

    # Measurement Jacobian matrix H
    def measurement_jacobian(x):
        return np.array([[1, 0, 0],
                         [0, 1, 0]])

    # Initialize Extended Kalman Filter
    num_states = 3
    dim_z = 2

    x = np.zeros((num_states, 1))
    P = np.eye(num_states)

    fused_data = []

    for i in range(min_length):
        # Prediction step
        F = np.array([[1, dt, 0],
                      [0, 1, dt],
                      [0, 0, 1]])

        x = state_transition_function(x)
        P = F @ P @ F.T

        # Update step
        y = np.array([[vision_data[i, 0]], [vision_data[i, 1]]]) - measurement_function(x)
        H = measurement_jacobian(x)

        S = H @ P @ H.T + R
        K = P @ H.T @ linalg.inv(S)
        x = x + K @ y
        P = (np.eye(num_states) - K @ H) @ P

        w_vision = K[0, 0]  # Weight for vision data
        w_magnetic = K[0, 0]  # Weight for magnetic data

        fused_value = w_vision * vision_data[i] + w_magnetic * magnetic_data[i]
        fused_data.append(fused_value)

    return np.array(fused_data)

vision_data_path = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\vision_EM_data_1june2023.csv"
magnetic_data_path = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\EMdata _zero deg.csv"

vision_data = pd.read_csv(vision_data_path)[["X", "Y"]].values
magnetic_data = pd.read_csv(magnetic_data_path)[["Pos(Mx)", "Pos(My)"]].values

R = np.eye(2)  # Measurement noise covariance matrix

fused_data = sensor_fusion(vision_data, magnetic_data)

plt.figure()
# Plotting the original vision and magnetic data
plt.plot(vision_data[:, 0], vision_data[:, 1], label='Vision Data')
plt.plot(magnetic_data[:, 0], magnetic_data[:, 1], label='Magnetic Data')

# Plotting the fused data
plt.plot(fused_data[:, 0], fused_data[:, 1], label='Fused Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sensor Fusion using Extended Kalman Filter')
plt.legend()
plt.show()

# Calculate average differences in X and Y
avg_x_vision = np.mean(fused_data[:, 0] - vision_data[:, 0])
avg_y_vision = np.mean(fused_data[:, 1] - vision_data[:, 1])

print("Average difference in X (Vision):", avg_x_vision)
print("Average difference in Y (Vision):", avg_y_vision)

avg_x_magnetic = np.mean(fused_data[:, 0] - magnetic_data[:, 0])
avg_y_magnetic = np.mean(fused_data[:, 1] - magnetic_data[:, 1])

print("Average difference in X (Magnetic):", avg_x_magnetic)
print("Average difference in Y (Magnetic):", avg_y_magnetic)
