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
combined_data = pd.concat([gps_data, imu_data], axis=1)

# Plot object trajectory before filtering
plt.figure(figsize=(12, 12))
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
H_lat = np.array([[1, 0, 0], [0, 0, 1]])  # Measurement matrix for latitude and acceleration in X-direction
H_lon = np.array([[1, 0, 0], [0, 0, 1]])  # Measurement matrix for longitude and acceleration in Y-direction
H_acc_x = np.array([[0, 1, 0]])  # Measurement matrix for acceleration in X-direction
H_acc_y = np.array([[0, 1, 0]])  # Measurement matrix for acceleration in Y-direction
H_acc_z = np.array([[0, 0, 1]])  # Measurement matrix for acceleration in Z-direction
H_ang_vel_x = np.array([[0, 1, 0]])  # Measurement matrix for angular velocity in X-direction
H_ang_vel_y = np.array([[0, 1, 0]])  # Measurement matrix for angular velocity in Y-direction
H_ang_vel_z = np.array([[0, 0, 1]])  # Measurement matrix for angular velocity in Z-direction
H_orient_x = np.array([[0, 1, 0]])  # Measurement matrix for orientation in X-direction
H_orient_y = np.array([[0, 1, 0]])  # Measurement matrix for orientation in Y-direction
H_orient_z = np.array([[0, 0, 1]])  # Measurement matrix for orientation in Z-direction
Q = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])  # Process noise covariance matrix
R_lat = np.array([[1, 0], [0, 1]])  # Measurement noise covariance matrix for latitude and acceleration in X-direction
R_lon = np.array([[1, 0], [0, 1]])  # Measurement noise covariance matrix for longitude and acceleration in Y-direction
R_acc_x = np.array([[1]])  # Measurement noise covariance matrix for acceleration in X-direction
R_acc_y = np.array([[1]])  # Measurement noise covariance matrix for acceleration in Y-direction
R_acc_z = np.array([[1]])  # Measurement noise covariance matrix for acceleration in Z-direction
R_ang_vel_x = np.array([[1]])  # Measurement noise covariance matrix for angular velocity in X-direction
R_ang_vel_y = np.array([[1]])  # Measurement noise covariance matrix for angular velocity in Y-direction
R_ang_vel_z = np.array([[1]])  # Measurement noise covariance matrix for angular velocity in Z-direction
R_orient_x = np.array([[1]])  # Measurement noise covariance matrix for orientation in X-direction
R_orient_y = np.array([[1]])  # Measurement noise covariance matrix for orientation in Y-direction
R_orient_z = np.array([[1]])  # Measurement noise covariance matrix for orientation in Z-direction
x = np.array([[0], [0], [0]])  # Initial state vector
P = np.eye(3)  # Initial state covariance matrix

# Kalman filter loop
filtered_latitude = []
filtered_longitude = []
filtered_acceleration_x = []
filtered_acceleration_y = []
filtered_acceleration_z = []
filtered_angular_velocity_x = []
filtered_angular_velocity_y = []
filtered_angular_velocity_z = []
filtered_orientation_x = []
filtered_orientation_y = []
filtered_orientation_z = []

for i in range(len(combined_data)):
    # Update matrices for each iteration
    if not np.isnan(gps_latitude[i]):
        H_lat[0, 0] = 1  # Update the first element of the measurement matrix for latitude
        R_lat[0, 0] = 0.1  # Update the measurement noise covariance for latitude
    else:
        H_lat[0, 0] = 0  # Set the first element of the measurement matrix for latitude to zero if there is no GPS measurement
        R_lat[0, 0] = 100  # Increase the measurement noise covariance for latitude if there is no GPS measurement

    if not np.isnan(gps_longitude[i]):
        H_lon[0, 0] = 1  # Update the first element of the measurement matrix for longitude
        R_lon[0, 0] = 0.1  # Update the measurement noise covariance for longitude
    else:
        H_lon[0, 0] = 0  # Set the first element of the measurement matrix for longitude to zero if there is no GPS measurement
        R_lon[0, 0] = 100  # Increase the measurement noise covariance for longitude if there is no GPS measurement

    if not np.isnan(imu_acceleration_x[i]):
        H_acc_x[0, 1] = 1  # Update the second element of the measurement matrix for acceleration in X-direction
        R_acc_x[0, 0] = 0.1  # Update the measurement noise covariance for acceleration in X-direction
    else:
        H_acc_x[0, 1] = 0  # Set the second element of the measurement matrix for acceleration in X-direction to zero if there is no IMU measurement
        R_acc_x[0, 0] = 100  # Increase the measurement noise covariance for acceleration in X-direction if there is no IMU measurement

    if not np.isnan(imu_acceleration_y[i]):
        H_acc_y[0, 1] = 1  # Update the second element of the measurement matrix for acceleration in Y-direction
        R_acc_y[0, 0] = 0.1  # Update the measurement noise covariance for acceleration in Y-direction
    else:
        H_acc_y[0, 1] = 0  # Set the second element of the measurement matrix for acceleration in Y-direction to zero if there is no IMU measurement
        R_acc_y[0, 0] = 100  # Increase the measurement noise covariance for acceleration in Y-direction if there is no IMU measurement

    if not np.isnan(imu_acceleration_z[i]):
        H_acc_z[0, 2] = 1  # Update the third element of the measurement matrix for acceleration in Z-direction
        R_acc_z[0, 0] = 0.1  # Update the measurement noise covariance for acceleration in Z-direction
    else:
        H_acc_z[0, 2] = 0  # Set the third element of the measurement matrix for acceleration in Z-direction to zero if there is no IMU measurement
        R_acc_z[0, 0] = 100  # Increase the measurement noise covariance for acceleration in Z-direction if there is no IMU measurement

    if not np.isnan(imu_angular_velocity_x[i]):
        H_ang_vel_x[0, 1] = 1  # Update the second element of the measurement matrix for angular velocity in X-direction
        R_ang_vel_x[0, 0] = 0.1  # Update the measurement noise covariance for angular velocity in X-direction
    else:
        H_ang_vel_x[0, 1] = 0  # Set the second element of the measurement matrix for angular velocity in X-direction to zero if there is no IMU measurement
        R_ang_vel_x[0, 0] = 100  # Increase the measurement noise covariance for angular velocity in X-direction if there is no IMU measurement

    if not np.isnan(imu_angular_velocity_y[i]):
        H_ang_vel_y[0, 1] = 1  # Update the second element of the measurement matrix for angular velocity in Y-direction
        R_ang_vel_y[0, 0] = 0.1  # Update the measurement noise covariance for angular velocity in Y-direction
    else:
        H_ang_vel_y[0, 1] = 0  # Set the second element of the measurement matrix for angular velocity in Y-direction to zero if there is no IMU measurement
        R_ang_vel_y[0, 0] = 100  # Increase the measurement noise covariance for angular velocity in Y-direction if there is no IMU measurement

    if not np.isnan(imu_angular_velocity_z[i]):
        H_ang_vel_z[0, 2] = 1  # Update the third element of the measurement matrix for angular velocity in Z-direction
        R_ang_vel_z[0, 0] = 0.1  # Update the measurement noise covariance for angular velocity in Z-direction
    else:
        H_ang_vel_z[0, 2] = 0  # Set the third element of the measurement matrix for angular velocity in Z-direction to zero if there is no IMU measurement
        R_ang_vel_z[0, 0] = 100  # Increase the measurement noise covariance for angular velocity in Z-direction if there is no IMU measurement

    if not np.isnan(imu_orientation_x[i]):
        H_orient_x[0, 1] = 1  # Update the second element of the measurement matrix for orientation in X-direction
        R_orient_x[0, 0] = 0.1  # Update the measurement noise covariance for orientation in X-direction
    else:
        H_orient_x[0, 1] = 0  # Set the second element of the measurement matrix for orientation in X-direction to zero if there is no IMU measurement
        R_orient_x[0, 0] = 100  # Increase the measurement noise covariance for orientation in X-direction if there is no IMU measurement

    if not np.isnan(imu_orientation_y[i]):
        H_orient_y[0, 1] = 1  # Update the second element of the measurement matrix for orientation in Y-direction
        R_orient_y[0, 0] = 0.1  # Update the measurement noise covariance for orientation in Y-direction
    else:
        H_orient_y[0, 1] = 0  # Set the second element of the measurement matrix for orientation in Y-direction to zero if there is no IMU measurement
        R_orient_y[0, 0] = 100  # Increase the measurement noise covariance for orientation in Y-direction if there is no IMU measurement

    if not np.isnan(imu_orientation_z[i]):
        H_orient_z[0, 2] = 1  # Update the third element of the measurement matrix for orientation in Z-direction
        R_orient_z[0, 0] = 0.1  # Update the measurement noise covariance for orientation in Z-direction
    else:
        H_orient_z[0, 2] = 0  # Set the third element of the measurement matrix for orientation in Z-direction to zero if there is no IMU measurement
        R_orient_z[0, 0] = 100  # Increase the measurement noise covariance for orientation in Z-direction if there is no IMU measurement

    # Prediction step
    x = np.dot(A, x)
    P = np.dot(A, np.dot(P, A.T)) + Q

    # Measurement update step
    if H_lat[0, 0] == 1:
        y_lat = np.array([[gps_latitude[i]], [imu_acceleration_x[i]]]) - np.dot(H_lat, x)
        S_lat = np.dot(H_lat, np.dot(P, H_lat.T)) + R_lat
        K_lat = np.dot(np.dot(P, H_lat.T), np.linalg.inv(S_lat))
        x = x + np.dot(K_lat, y_lat)
        P = np.dot((np.eye(3) - np.dot(K_lat, H_lat)), P)

    if H_lon[0, 0] == 1:
        y_lon = np.array([[gps_longitude[i]], [imu_acceleration_y[i]]]) - np.dot(H_lon, x)
        S_lon = np.dot(H_lon, np.dot(P, H_lon.T)) + R_lon
        K_lon = np.dot(np.dot(P, H_lon.T), np.linalg.inv(S_lon))
        x = x + np.dot(K_lon, y_lon)
        P = np.dot((np.eye(3) - np.dot(K_lon, H_lon)), P)

    if H_acc_x[0, 1] == 1:
        y_acc_x = imu_acceleration_x[i] - np.dot(H_acc_x, x)[0, 0]
        S_acc_x = np.dot(H_acc_x, np.dot(P, H_acc_x.T)) + R_acc_x
        K_acc_x = np.dot(np.dot(P, H_acc_x.T), np.linalg.inv(S_acc_x))
        x = x + np.dot(K_acc_x, y_acc_x)
        P = np.dot((np.eye(3) - np.dot(K_acc_x, H_acc_x)), P)

    if H_acc_y[0, 1] == 1:
        y_acc_y = imu_acceleration_y[i] - np.dot(H_acc_y, x)[0, 0]
        S_acc_y = np.dot(H_acc_y, np.dot(P, H_acc_y.T)) + R_acc_y
        K_acc_y = np.dot(np.dot(P, H_acc_y.T), np.linalg.inv(S_acc_y))
        x = x + np.dot(K_acc_y, y_acc_y)
        P = np.dot((np.eye(3) - np.dot(K_acc_y, H_acc_y)), P)

    if H_acc_z[0, 2] == 1:
        y_acc_z = imu_acceleration_z[i] - np.dot(H_acc_z, x)[0, 0]
        S_acc_z = np.dot(H_acc_z, np.dot(P, H_acc_z.T)) + R_acc_z
        K_acc_z = np.dot(np.dot(P, H_acc_z.T), np.linalg.inv(S_acc_z))
        x = x + np.dot(K_acc_z, y_acc_z)
        P = np.dot((np.eye(3) - np.dot(K_acc_z, H_acc_z)), P)

    if H_ang_vel_x[0, 1] == 1:
        y_ang_vel_x = imu_angular_velocity_x[i] - np.dot(H_ang_vel_x, x)[0, 0]
        S_ang_vel_x = np.dot(H_ang_vel_x, np.dot(P, H_ang_vel_x.T)) + R_ang_vel_x
        K_ang_vel_x = np.dot(np.dot(P, H_ang_vel_x.T), np.linalg.inv(S_ang_vel_x))
        x = x + np.dot(K_ang_vel_x, y_ang_vel_x)
        P = np.dot((np.eye(3) - np.dot(K_ang_vel_x, H_ang_vel_x)), P)

    if H_ang_vel_y[0, 1] == 1:
        y_ang_vel_y = imu_angular_velocity_y[i] - np.dot(H_ang_vel_y, x)[0, 0]
        S_ang_vel_y = np.dot(H_ang_vel_y, np.dot(P, H_ang_vel_y.T)) + R_ang_vel_y
        K_ang_vel_y = np.dot(np.dot(P, H_ang_vel_y.T), np.linalg.inv(S_ang_vel_y))
        x = x + np.dot(K_ang_vel_y, y_ang_vel_y)
        P = np.dot((np.eye(3) - np.dot(K_ang_vel_y, H_ang_vel_y)), P)

    if H_ang_vel_z[0, 2] == 1:
        y_ang_vel_z = imu_angular_velocity_z[i] - np.dot(H_ang_vel_z, x)[0, 0]
        S_ang_vel_z = np.dot(H_ang_vel_z, np.dot(P, H_ang_vel_z.T)) + R_ang_vel_z
        K_ang_vel_z = np.dot(np.dot(P, H_ang_vel_z.T), np.linalg.inv(S_ang_vel_z))
        x = x + np.dot(K_ang_vel_z, y_ang_vel_z)
        P = np.dot((np.eye(3) - np.dot(K_ang_vel_z, H_ang_vel_z)), P)

    if H_orient_x[0, 1] == 1:
        y_orient_x = imu_orientation_x[i] - np.dot(H_orient_x, x)[0, 0]
        S_orient_x = np.dot(H_orient_x, np.dot(P, H_orient_x.T)) + R_orient_x
        K_orient_x = np.dot(np.dot(P, H_orient_x.T), np.linalg.inv(S_orient_x))
        x = x + np.dot(K_orient_x, y_orient_x)
        P = np.dot((np.eye(3) - np.dot(K_orient_x, H_orient_x)), P)

    if H_orient_y[0, 1] == 1:
        y_orient_y = imu_orientation_y[i] - np.dot(H_orient_y, x)[0, 0]
        S_orient_y = np.dot(H_orient_y, np.dot(P, H_orient_y.T)) + R_orient_y
        K_orient_y = np.dot(np.dot(P, H_orient_y.T), np.linalg.inv(S_orient_y))
        x = x + np.dot(K_orient_y, y_orient_y)
        P = np.dot((np.eye(3) - np.dot(K_orient_y, H_orient_y)), P)

    if H_orient_z[0, 2] == 1:
        y_orient_z = imu_orientation_z[i] - np.dot(H_orient_z, x)[0, 0]
        S_orient_z = np.dot(H_orient_z, np.dot(P, H_orient_z.T)) + R_orient_z
        K_orient_z = np.dot(np.dot(P, H_orient_z.T), np.linalg.inv(S_orient_z))
        x = x + np.dot(K_orient_z, y_orient_z)
        P = np.dot((np.eye(3) - np.dot(K_orient_z, H_orient_z)), P)

    # Store the filtered values
    filtered_latitude.append(x[0, 0])
    filtered_longitude.append(x[1, 0])
    filtered_acceleration_x.append(x[2, 0])
    filtered_acceleration_y.append(x[2, 0])
    filtered_acceleration_z.append(x[2, 0])
    filtered_angular_velocity_x.append(x[2, 0])
    filtered_angular_velocity_y.append(x[2, 0])
    filtered_angular_velocity_z.append(x[2, 0])
    filtered_orientation_x.append(x[2, 0])
    filtered_orientation_y.append(x[2, 0])
    filtered_orientation_z.append(x[2, 0])

# Print the filtered values
print("Filtered Latitude:", filtered_latitude)
print("Filtered Longitude:", filtered_longitude)
print("Filtered Acceleration X:", filtered_acceleration_x)
print("Filtered Acceleration Y:", filtered_acceleration_y)
print("Filtered Acceleration Z:", filtered_acceleration_z)
print("Filtered Angular Velocity X:", filtered_angular_velocity_x)
print("Filtered Angular Velocity Y:", filtered_angular_velocity_y)
print("Filtered Angular Velocity Z:", filtered_angular_velocity_z)
print("Filtered Orientation X:", filtered_orientation_x)
print("Filtered Orientation Y:", filtered_orientation_y)
print("Filtered Orientation Z:", filtered_orientation_z)
