import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, initial_state):
        self.state = initial_state
        self.covariance = np.eye(len(initial_state))
        self.process_noise = np.eye(len(initial_state))
    
    def predict(self, F, B, u):
        # Prediction step
        predicted_state = F @ self.state + B @ u
        predicted_covariance = F @ self.covariance @ F.T + self.process_noise
        return predicted_state, predicted_covariance
    
    def update(self, measurement, H, R):
        # Update step
        residual = measurement - H @ self.state
        residual_covariance = H @ self.covariance @ H.T + R
        kalman_gain = self.covariance @ H.T @ np.linalg.inv(residual_covariance)
        self.state += kalman_gain @ residual
        self.covariance = (np.eye(len(self.state)) - kalman_gain @ H) @ self.covariance

# Define the sonar data
latitude = 35.6789
longitude = -120.5432
depth = 50
object_detected = "Shipwreck"
sonar_system = "Multibeam sonar"

# Define the acoustic modem data
latitude_modem = 35.6795
longitude_modem = -120.5421
depth_modem = 55
modem_id = "M1"
range_to_surface_station = 800

# Define the IMU data
roll = 10.5
pitch = -2.3
yaw = 45.1
velocity = 0.8
acceleration = 2.2

# Define initial state
initial_state = np.array([[latitude], [longitude], [depth], [roll], [pitch], [yaw], [velocity], [acceleration]])

# Initialize the Kalman filter
kf = KalmanFilter(initial_state)

# Define system matrices
F = np.eye(len(initial_state))
B = np.zeros((len(initial_state), 3))
u = np.array([[roll], [pitch], [yaw]])  # Control input (IMU data)


# Sound signal parameters
distance = 200  # Distance between Tx and Rx (in meters)
depth_sound = depth  # Depth of the sound signal transmission
data_rate = 64  # Data rate in bits/sec

# Generate sound signals
time = np.linspace(0, distance / velocity, int(distance / velocity * data_rate))
sound_signal_before_filter = np.sin(2 * np.pi * data_rate * time)

# Add noise to the sound signal
noise_level = 0.2  # Adjust the noise level as desired
noisy_sound_signal = sound_signal_before_filter + np.random.normal(0, noise_level, size=len(time))

# Apply the Kalman filter to reduce the noise
filtered_sound_signal = np.zeros_like(noisy_sound_signal)
filtered_sound_signal[0] = noisy_sound_signal[0]

# Define measurement matrices
H = np.eye(len(time), len(initial_state))
R = np.eye(len(time)) * noise_level  # Measurement noise covariance

# Apply the Kalman filter to reduce the noise
filtered_sound_signal = np.zeros_like(noisy_sound_signal)
filtered_sound_signal[0] = noisy_sound_signal[0]

for i in range(1, len(time)):
    predicted_state, predicted_covariance = kf.predict(F, B, u)
    measurement = np.array([[noisy_sound_signal[i]]])
    kf.update(measurement, H, R)
    filtered_sound_signal[i] = kf.state[0]

# Plot the sound signal with noise and the filtered sound signal
plt.figure(figsize=(10, 6))

# Original sound signal
plt.subplot(2, 1, 1)
plt.plot(time, sound_signal_before_filter, 'b', label="Original sound signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# Noisy sound signal and filtered sound signal
plt.subplot(2, 1, 2)
plt.plot(time, noisy_sound_signal, 'r', label="Sound signal with noise")
plt.plot(time, filtered_sound_signal, 'g', label="Filtered sound signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()

