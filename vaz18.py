import pandas as pd
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
import matplotlib.pyplot as plt


# File path of the dataset
file_path = r"C:\Users\SHAIK VAZEER AHAMED\Downloads\AUV- Snapir\Tedelyne DVL\teledyne_navigator_measurements.xlsx"


# Load dataset
data = pd.read_excel(file_path)


# Convert non-numeric values to NaN
data[['latitude', 'longitute', 'Errror [m/sec]', 'w speed [m/sec]', 'v speed [m/sec]', 'u speed [m/sec]']] = data[['latitude', 'longitute', 'Errror [m/sec]', 'w speed [m/sec]', 'v speed [m/sec]', 'u speed [m/sec]']].apply(pd.to_numeric, errors='coerce')


# Define the ExtendedKalmanFilter class with the overridden predict method
class CustomExtendedKalmanFilter(ExtendedKalmanFilter):
    def predict(self, fx):
        """Perform the predict step of the Kalman filter with a custom system dynamics function."""
        self.x = fx(self.x)  # Update the state using the custom system dynamics function
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q  # Update the covariance matrix

        
# Create Extended Kalman Filter
num_states = 6  # Number of states to track (u speed, v speed, w speed, latitude, longitude, error)
dim_z = 3  # Number of measurements (u speed, v speed, w speed)

ekf = CustomExtendedKalmanFilter(dim_x=num_states, dim_z=dim_z)


# Define the system dynamics function (transition matrix) for the EKF
def system_dynamics(x):
    # Extract the states
    latitude = x[0]
    longitude = x[1]
    u_speed = x[2]
    v_speed = x[3]
    w_speed = x[4]
    error = x[5]

    # Modify the states based on the system dynamics equations
    dt = 1.0  # Adjust the time step as needed
    u_speed += 0.1 * np.random.randn() * dt + 0.01  # Example nonlinear dynamics
    v_speed += 0.1 * np.random.randn() * dt + 0.01  # Example nonlinear dynamics
    w_speed += 0.1 * np.random.randn() * dt + 0.01  # Example nonlinear dynamics

    # Return the updated states
    return np.array([latitude, longitude, u_speed, v_speed, w_speed, error])


# Define the measurement function for the EKF
def measurement_function(x):
    # Extract the relevant states for measurement
    u_speed = x[2]
    v_speed = x[3]
    w_speed = x[4]

    # Example nonlinear measurement equations
    measured_u_speed = u_speed + 0.1 * np.random.randn()
    measured_v_speed = v_speed + 0.1 * np.random.randn()
    measured_w_speed = w_speed + 0.1 * np.random.randn()

    # Return the measured values
    return np.array([measured_u_speed, measured_v_speed, measured_w_speed])


# Define the Jacobian of the measurement function
def measurement_jacobian(x):
    # Extract the relevant states for measurement
    u_speed = x[2]
    v_speed = x[3]
    w_speed = x[4]

    # Example nonlinear measurement Jacobian
    H = np.eye(dim_z, num_states)
    return H


# Initialize EKF
initial_state = np.array([data['latitude'].iloc[0], data['longitute'].iloc[0], data['u speed [m/sec]'].iloc[0], data['v speed [m/sec]'].iloc[0], data['w speed [m/sec]'].iloc[0], data['Errror [m/sec]'].iloc[0]])
initial_covariance = np.eye(num_states)

ekf.x = initial_state
ekf.P = initial_covariance

# Define the process noise covariance matrix
process_noise = np.eye(num_states) * 0.01  # Adjust the process noise covariance as needed
ekf.Q = process_noise

# Define the measurement noise covariance matrix
measurement_noise = np.eye(dim_z) * 0.01  # Adjust the measurement noise covariance as needed
ekf.R = measurement_noise

# Track the object using the Extended Kalman Filter
filtered_data = []
for _, row in data.iterrows():
    measurement = np.array([row['u speed [m/sec]'], row['v speed [m/sec]'], row['w speed [m/sec]']])

    # Perform the prediction step using the custom system dynamics function
    ekf.predict(fx=system_dynamics)

    # Perform the update step
    ekf.update(z=measurement, HJacobian=measurement_jacobian, Hx=measurement_function)

    # Save the estimated state
    filtered_data.append(ekf.x[:3])

filtered_data = np.array(filtered_data)


# Convert time to string representation for plotting
data['Time'] = data['Time'].apply(lambda t: t.strftime('%H:%M:%S'))


# Plot the original and filtered data
plt.figure(figsize=(12, 6))

# Plot original data
plt.subplot(1, 2, 1)
plt.plot(data['Time'], data['u speed [m/sec]'], label='Original u speed')
plt.plot(data['Time'], data['v speed [m/sec]'], label='Original v speed')
plt.plot(data['Time'], data['w speed [m/sec]'], label='Original w speed')
plt.xlabel('Time')
plt.ylabel('Velocity [m/sec]')
plt.title('Original Data')
plt.legend()

# Plot filtered data
plt.subplot(1, 2, 2)
plt.plot(data['Time'], filtered_data[:, 0], label='Filtered u speed')
plt.plot(data['Time'], filtered_data[:, 1], 'bo-', label='Filtered v speed')
plt.plot(data['Time'], filtered_data[:, 2], 'r*-',label='Filtered w speed')
plt.xlabel('Time')
plt.ylabel('Velocity [m/sec]')
plt.title('Filtered Data')
plt.legend()

plt.tight_layout()
plt.show()