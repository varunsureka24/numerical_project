import numpy as np
import matplotlib.pyplot as plt
from beam import t, x, dt

# ── 1. The FFT algorithm from scratch ───────────────────────────────────────
# This is the Cooley-Tukey Radix-2 algorithm.
# It takes a signal and returns its frequency content.
def my_fft(signal):
    N = len(signal)

    # Base case: if only one element, return it as-is
    if N == 1:
        return signal

    # FFT works best when the signal length is a power of 2
    # so we'll enforce that before calling this function (see Step 2)

    # Split the signal into even and odd indexed elements
    even = my_fft(signal[0::2])
    odd  = my_fft(signal[1::2])

    # Compute the "twiddle factors" — complex exponentials that
    # combine the even and odd halves
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]

    # Combine into the final result
    return [even[k] + T[k] for k in range(N // 2)] + \
           [even[k] - T[k] for k in range(N // 2)]


# ── 2. Prepare the signal ───────────────────────────────────────────────────
# The FFT requires the signal length to be a power of 2.
# We trim our signal down to the nearest power of 2.
N_original = len(x)
N = 2 ** int(np.floor(np.log2(N_original)))  # nearest power of 2
x_trimmed = x[:N]                             # trim the signal

print(f"Original signal length: {N_original}")
print(f"Trimmed to power of 2:  {N}")

# ── 3. Run our FFT ──────────────────────────────────────────────────────────
X_ours = np.array(my_fft(list(x_trimmed)))

# ── 4. Run NumPy's built-in FFT to verify ───────────────────────────────────
X_numpy = np.fft.fft(x_trimmed)

# ── 5. Compute the frequency axis ───────────────────────────────────────────
# Each index in the FFT output corresponds to a specific frequency.
# We only look at the first half (the second half is a mirror image).
freqs = np.fft.fftfreq(N, d=dt)   # frequency values in Hz
half  = N // 2                     # only need the positive frequencies

# ── 6. Compute the magnitude of each frequency component ────────────────────
magnitude_ours  = np.abs(X_ours[:half])
magnitude_numpy = np.abs(X_numpy[:half])

# ── 7. Plot and compare ─────────────────────────────────────────────────────
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(freqs[:half], magnitude_ours, color='steelblue')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Our FFT")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(freqs[:half], magnitude_numpy, color='tomato')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("NumPy's FFT (verification)")
plt.grid(True)

plt.tight_layout()
plt.xlim([0, 60])  # focus on the low-frequency range where our modes are
plt.show()

# ── 8. Find and print the dominant frequency ────────────────────────────────
peak_idx = np.argmax(magnitude_ours)
peak_freq = freqs[peak_idx]
print(f"\nDominant frequency detected by our FFT: {peak_freq:.2f} Hz")
print(f"Expected Mode 1 frequency:              2.58 Hz")