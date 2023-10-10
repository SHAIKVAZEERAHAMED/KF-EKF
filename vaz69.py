import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt


# Kalman Filter Sensor Fusion

def kalman_filter(z, A, H, Q, R, x0, P0):
    """
    Performs sensor fusion using Kalman Filter algorithm.

    Arguments:
    - z: Concatenated measurement vector (numpy array)
    - A: State transition matrix
    - H: Observation matrix
    - Q: Process noise covariance
    - R: Measurement noise covariance
    - x0: Initial state estimate
    - P0: Initial covariance estimate

    Returns:
    - x: Fused state estimate
    - P: Updated covariance estimate
    """
    n = len(A[0])  # Dimension of the state vector
    m = len(H[0])  # Dimension of the measurement vector

    # Initialize state estimate and covariance estimate
    x = x0
    P = P0

    fused_states = [x[0]]

    for measurement in z:
        # Prediction step
        x = A @ x
        P = A @ P @ A.T + Q

        # Kalman gain
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        # Update step
        y = measurement - H @ x
        x += K @ y
        P = (np.eye(n) - K @ H) @ P

        fused_states.append(x[0])

    return fused_states, P


# Example usage

# Two separate temperature measurements
z1 = np.array([25.5, 26.0, 25.8, 26.2])  # From sensor 1
z2 = np.array([26.2, 26.5, 26.8, 27.0])  # From sensor 2

# Concatenate the measurements
z = np.concatenate([z1, z2])

# Define the system matrices
A = np.eye(2)  # State transition matrix
H = np.array([[1, 0]])  # Observation matrix (modified)

# Process noise covariance
Q = block_diag(0.1, 0.1)  # Assuming uncorrelated process noises

# Measurement noise covariance
R = block_diag(0.2)  # Assuming uncorrelated measurement noises (modified)

# Initial state estimate
x0 = np.array([0, 0])  # Assuming initial temperature estimates are zero

# Initial covariance estimate
P0 = np.eye(2)  # Assuming perfect initial estimates

# Perform sensor fusion using Kalman Filter
fused_temperatures, P_fused = kalman_filter(z, A, H, Q, R, x0, P0)

# Assuming 4 fused temperature estimates
timestamps = ['09:00', '09:15', '09:30', '09:45','10:00', '10:15', '10:30', '10:45','11:00']

plt.plot(timestamps[:4], z1, label='Sensor 1')
plt.plot(timestamps[:4], z2, label='Sensor 2')
plt.plot(timestamps[:9], fused_temperatures, label='Fused')



plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.title('Fused Temperature Estimates')
plt.legend()

plt.show()
