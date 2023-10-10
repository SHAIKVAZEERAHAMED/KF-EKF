import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# # SNR parameters
# snr_max = -214.59405337517438  # Maximum SNR in dB
# snr_min = -251.25880166061785   # Minimum SNR in dB
# snr_step = -2  # SNR step size in dB

# snr_max = -84.82905759315307  # Maximum SNR in dB
# snr_min = -121.49380587859653  # Minimum SNR in dB
# snr_step = -2  # SNR step size in dB

# snr_max = -45.13008821749291  # Maximum SNR in dB
# snr_min = -74.8205815973262  # Minimum SNR in dB
# snr_step = -2  # SNR step size in dB

snr_max = 14  # Maximum SNR in dB
snr_min = -32 # Minimum SNR in dB
snr_step = -2  # SNR step size in dB

# Generate SNR values
snr_values = np.arange(snr_max,snr_min+ snr_step , snr_step)

ber_values = 0.5 * erfc(np.sqrt(10 ** (snr_values / 10) / 2))
print(np.sqrt(10 ** (snr_values / 10)))
print(0.5 * erfc(np.sqrt(10 ** (snr_values / 10) / 2)))
# Plot BER vs SNR
plt.semilogy(snr_values, ber_values, marker='o')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('BER vs SNR')
plt.grid(True)

# Automatically adjust the y-axis limits based on the range of BER values
plt.ylim(bottom=np.min(ber_values) / 10, top=1)

plt.show()