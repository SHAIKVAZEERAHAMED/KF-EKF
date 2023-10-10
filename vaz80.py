import pandas as pd
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
import matplotlib.pyplot as plt

file_path = r"C:\Users\SHAIK VAZEER AHAMED\OneDrive\Desktop\csv_kf_ekf\EMdata -30deg.csv"

data = pd.read_csv(file_path)

num_states = 3 
dim_z = 2  

ekf = ExtendedKalmanFilter(dim_x=num_states, dim_z=dim_z)

initial_state = np.array([data['Pos(Mx)'].iloc[0], data['Pos(My)'].iloc[0], 0.0])
initial_covariance = np.eye(num_states) * 0.1  

ekf.x = initial_state
ekf.P = initial_covariance

def state_transition_function(x, dt):
    F = np.array([[1, dt, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    return np.dot(F, x)

def measurement_function(x):
    return np.array([x[0], x[1]])

def measurement_jacobian(x):
    return np.array([[1, 0, 0], [0, 1, 0]])

ekf.f = state_transition_function
ekf.h = measurement_function
ekf.H = measurement_jacobian

process_noise = np.eye(num_states) * 0.01  
ekf.Q = process_noise

measurement_noise = np.eye(dim_z) * 0.1  
ekf.R = measurement_noise

estimated_values = []
for _, row in data.iterrows():
    measurement = np.array([
        row['Pos(Mx)'],
        row['Pos(My)']
    ])

    # Predict the object's state
    ekf.predict()

    # Update the object's state based on the measurement
    ekf.update(measurement,measurement_jacobian,measurement_function)

    # Save the estimated state
    estimated_values.append(ekf.x.copy())

estimated_values = np.array(estimated_values)

# Plot the original and estimated values
plt.figure()
plt.plot(data['Pos(Mx)'], data['Pos(My)'], 'g*-', label='Original')
plt.plot(estimated_values[:, 0], estimated_values[:, 1], 'y*-', label='Estimated')

plt.xlabel('Position X')
plt.ylabel('Position Y')
plt.title('Original and Estimated Positions')
plt.legend()
plt.show()

# Calculate average differences in X and Y
avg_x = np.mean(estimated_values[:, 0] - data['Pos(Mx)'])
avg_y = np.mean(estimated_values[:, 1] - data['Pos(My)'])

print("Average difference in X:", avg_x)
print("Average difference in Y:", avg_y)
