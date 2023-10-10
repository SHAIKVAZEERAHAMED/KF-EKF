import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load IMU and GPS datasets
imu_data = pd.read_csv(r'C:\Users\SHAIK VAZEER AHAMED\Downloads\philos_2019_10_29__16_00_19_target_and_sailboats\philos_2019_10_29__16_00_19_target_and_sailboats\csv\imu.csv')
gps_data = pd.read_csv(r'C:\Users\SHAIK VAZEER AHAMED\Downloads\philos_2019_10_29__16_00_19_target_and_sailboats\philos_2019_10_29__16_00_19_target_and_sailboats\csv\gps.csv')

# Remove leading and trailing spaces from column names
imu_data.columns = imu_data.columns.str.strip()
gps_data.columns = gps_data.columns.str.strip()

# Extract relevant columns from datasets
imu_timestamps = imu_data['Timestamp']
imu_acceleration_x = imu_data['Linear acceleration x']
imu_acceleration_y = imu_data['Linear acceleration y']
imu_acceleration_z = imu_data['Linear acceleration z']
imu_angular_velocity_x = imu_data['Angular velocity x']
imu_angular_velocity_y = imu_data['Angular velocity y']
imu_angular_velocity_z = imu_data['Angular velocity z']
imu_orientation_x = imu_data['Orientation x']
imu_orientation_y = imu_data['Orientation y']
imu_orientation_z = imu_data['Orientation z']
gps_latitude = gps_data['latitude']
gps_longitude = gps_data['longitude']

# Combine GPS and IMU data

plt.figure()
# Plot object trajectory before filtering
plt.subplot(2, 2, 1)
plt.plot(gps_longitude, gps_latitude, label='Before Filtering')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.title('Object Trajectory - Before Filtering')

plt.subplot(2, 2, 2)
plt.plot(imu_timestamps, imu_acceleration_x, label='Acceleration X')
plt.plot(imu_timestamps, imu_acceleration_y, label='Acceleration Y')
plt.plot(imu_timestamps, imu_acceleration_z, label='Acceleration Z')
plt.xlabel('Timestamp')
plt.ylabel('Acceleration')
plt.legend()
plt.title('IMU - Linear Acceleration')

plt.subplot(2, 2, 3)
plt.plot(imu_timestamps, imu_angular_velocity_x, label='Angular Velocity X')
plt.plot(imu_timestamps, imu_angular_velocity_y, label='Angular Velocity Y')
plt.plot(imu_timestamps, imu_angular_velocity_z, label='Angular Velocity Z')
plt.xlabel('Timestamp')
plt.ylabel('Angular Velocity')
plt.legend()
plt.title('IMU - Angular Velocity')

plt.subplot(2, 2, 4)
plt.plot(imu_timestamps, imu_orientation_x, label='Orientation X')
plt.plot(imu_timestamps, imu_orientation_y, label='Orientation Y')
plt.plot(imu_timestamps, imu_orientation_z, label='Orientation Z')
plt.xlabel('Timestamp')
plt.ylabel('Orientation')
plt.legend()
plt.title('IMU - Orientation')

plt.show()

# Kalman filter initialization
dt = 1.0  # Time step
A = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])  # State transition matrix
H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Measurement matrix
Q = np.eye(3) * 0.1  # Process noise covariance
R = np.eye(3) * 1.0  # Measurement noise covariance

# Initialize lists for filtered states and measurements
filtered_states = []
filtered_acceleration_x = []
filtered_acceleration_y = []
filtered_acceleration_z = []
filtered_velocity_x = []
filtered_velocity_y = []
filtered_velocity_z = []
filtered_orientation_x = []
filtered_orientation_y = []
filtered_orientation_z = []


# Initialize state variables for each measurement
x_acc = np.array([imu_acceleration_x[0], imu_acceleration_y[0], imu_acceleration_z[0]])  # Initial state for acceleration
x_vel = np.array([imu_angular_velocity_x[0], imu_angular_velocity_y[0], imu_angular_velocity_z[0]])  # Initial state for angular velocity
x_ori = np.array([imu_orientation_x[0], imu_orientation_y[0], imu_orientation_z[0]])  # Initial state for orientation
P_acc = np.eye(3)  # Initial covariance for acceleration
P_vel = np.eye(3)  # Initial covariance for angular velocity
P_ori = np.eye(3)  # Initial covariance for orientation


# Apply Kalman filter using IMU measurements
# ... (previous code)

# Initialize state variables for acceleration, velocity, and orientation
x_acc = np.array([imu_acceleration_x[0], imu_acceleration_y[0], imu_acceleration_z[0]])
x_vel = np.array([imu_angular_velocity_x[0],imu_angular_velocity_y[0], imu_angular_velocity_z[0]])
x_ori = np.array([imu_orientation_x[0], imu_orientation_y[0], imu_orientation_z[0]])
P_acc = np.eye(3)
P_vel = np.eye(3)
P_ori = np.eye(3)

for i in range(len(imu_data)):
    # Prediction step for acceleration
    x_acc = np.dot(A, x_acc)
    P_acc = np.dot(np.dot(A, P_acc), A.T) + Q

    # Measurement update step for acceleration
    y_acc = np.array([imu_acceleration_x[i], imu_acceleration_y[i], imu_acceleration_z[i]])
    S_acc = np.dot(np.dot(H, P_acc), H.T) + R
    K_acc = np.dot(np.dot(P_acc, H.T), np.linalg.inv(S_acc))

    x_acc = x_acc + np.dot(K_acc, y_acc - np.dot(H, x_acc))
    P_acc = np.dot((np.eye(3) - np.dot(K_acc, H)), P_acc)

    # Prediction step for velocity
    x_vel = np.dot(A, x_vel)
    P_vel = np.dot(np.dot(A, P_vel), A.T) + Q

    # Measurement update step for velocity
    y_vel = np.array([imu_angular_velocity_x[i], imu_angular_velocity_y[i], imu_angular_velocity_z[i]])
    S_vel = np.dot(np.dot(H, P_vel), H.T) + R
    K_vel = np.dot(np.dot(P_vel, H.T), np.linalg.inv(S_vel))

    x_vel = x_vel + np.dot(K_vel, y_vel - np.dot(H, x_vel))
    P_vel = np.dot((np.eye(3) - np.dot(K_vel, H)), P_vel)

    # Prediction step for orientation
    x_ori = np.dot(A, x_ori)
    P_ori = np.dot(np.dot(A, P_ori), A.T) + Q

    # Measurement update step for orientation
    y_ori = np.array([imu_orientation_x[i], imu_orientation_y[i], imu_orientation_z[i]])
    S_ori = np.dot(np.dot(H, P_ori), H.T) + R
    K_ori = np.dot(np.dot(P_ori, H.T), np.linalg.inv(S_ori))

    x_ori = x_ori + np.dot(K_ori, y_ori - np.dot(H, x_ori))
    P_ori = np.dot((np.eye(3) - np.dot(K_ori, H)), P_ori)

    # Store the filtered values
    filtered_acceleration_x.append(x_acc[0])
    filtered_acceleration_y.append(x_acc[1])
    filtered_acceleration_z.append(x_acc[2])
    filtered_velocity_x.append(x_vel[0])
    filtered_velocity_y.append(x_vel[1])
    filtered_velocity_z.append(x_vel[2])
    filtered_orientation_x.append(x_ori[0])
    filtered_orientation_y.append(x_ori[1])
    filtered_orientation_z.append(x_ori[2])

# ... (rest of the code)


plt.figure()
# Plot object trajectory after filtering
plt.subplot(2, 2, 1)
plt.plot(gps_longitude, gps_latitude, label='After Filtering')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.title('Object Trajectory - After Filtering')

plt.subplot(2, 2, 2)
plt.plot(imu_timestamps, filtered_acceleration_x, label='Filtered Acceleration X')
plt.plot(imu_timestamps, filtered_acceleration_y, label='Filtered Acceleration Y')
plt.plot(imu_timestamps, filtered_acceleration_z, label='Filtered Acceleration Z')
plt.xlabel('Timestamp')
plt.ylabel('Acceleration')
plt.legend()
plt.title('Filtered IMU - Linear Acceleration')

plt.subplot(2, 2, 3)
plt.plot(imu_timestamps, filtered_velocity_x, label='Filtered Velocity X')
plt.plot(imu_timestamps, filtered_velocity_y, label='Filtered Velocity Y')
plt.plot(imu_timestamps, filtered_velocity_z, label='Filtered Velocity Z')
plt.xlabel('Timestamp')
plt.ylabel('Velocity')
plt.legend()
plt.title('Filtered IMU - Angular Velocity')

plt.subplot(2, 2, 4)
plt.plot(imu_timestamps, filtered_orientation_x, label='Filtered Orientation X')
plt.plot(imu_timestamps, filtered_orientation_y, label='Filtered Orientation Y')
plt.plot(imu_timestamps, filtered_orientation_z, label='Filtered Orientation Z')
plt.xlabel('Timestamp')
plt.ylabel('Orientation')
plt.legend()
plt.title('Filtered IMU - Orientation')

plt.show()
