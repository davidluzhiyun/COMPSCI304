import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

DEFAULT_WINDOW_TYPE = 'hamming'

DEFAULT_PREEMPHASIS_COEFFICIENT = 0.95

HOP_TIME = 0.01

FRAME_TIME = 0.02


# Read the .wav File
def read_wav(file_path):
    sr, data = scipy.io.wavfile.read(file_path)
    return sr, data


# Pre-emphasis
def preemphasis(signal, coefficient=DEFAULT_PREEMPHASIS_COEFFICIENT):
    emphasized_signal = np.append(signal[0], signal[1:] - coefficient * signal[:-1])
    return emphasized_signal


# Windowing
def apply_window(signal, window_type=DEFAULT_WINDOW_TYPE):
    window = scipy.signal.get_window(window_type, len(signal))
    windowed_signal = signal * window
    return windowed_signal


# Divide into 20ms frames
def divide_into_frames(signal, frame_size, hop_size):
    frames = []
    for i in range(0, len(signal) - frame_size + 1, hop_size):
        frame = signal[i:i + frame_size]
        frames.append(frame)
    return np.array(frames)


# Zero Padding
def zero_padding(signal, target_length):
    assert len(signal) <= target_length
    padded_signal = np.pad(signal, ((0, 0), (0, target_length - len(signal[0]))))
    return padded_signal


# FFT and truncate
def calculate_dft(signal):
    truncated_length = int(len(signal[0])/2) + 1
    power_spectrum = np.abs(np.fft.fft(signal))[:, :truncated_length]
    return power_spectrum


# Mel Filter Banks
def mel_filter_bank(sr, n_fft, n_mels):
    mel_frequencies = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_points = 2595 * np.log10(1 + mel_frequencies / 700)

    # Convert mel points to frequency bins
    bin_points = np.floor((n_fft + 1) * mel_points / sr).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1))

    for i in range(1, n_mels + 1):
        filters[i - 1, bin_points[i - 1]:bin_points[i]] = (bin_points[i] - mel_points[i - 1]) / (
                    bin_points[i] - bin_points[i - 1])
        filters[i - 1, bin_points[i]:bin_points[i + 1]] = (mel_points[i + 1] - bin_points[i]) / (
                    bin_points[i + 1] - bin_points[i])

    return filters.T


# Step 9: Log Spectra
def calculate_log_spectra(magnitude_spectrum):
    return 20 * np.log10(magnitude_spectrum)


# Step 10: Cepstral Coefficients
def calculate_cepstra(log_spectra, num_cepstral_coefficients=13):
    return scipy.fftpack.dct(log_spectra, axis=0, type=2, norm='ortho')[:num_cepstral_coefficients]


# Plot audio wave form
def plot_audio(sr, data):
    length = data.shape[0] / sr
    time = np.linspace(0., length, data.shape[0])
    plt.plot(time, data)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()


# Plot power frequency graphs
def plot_power(max_frequency, data):
    frequency = np.linspace(0., max_frequency, data.shape[0])
    plt.plot(frequency, data)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power")
    plt.show()


# Main Code
FILE_PATH = 'one.wav'
SR, DATA = read_wav(FILE_PATH)

# Plotting the file being read
plot_audio(SR, DATA)

# Pre-emphasis
emphasized_audio = preemphasis(DATA)

plot_audio(SR, emphasized_audio)

# To frame
frame_size = int(FRAME_TIME * SR)
hop_size = int(HOP_TIME * SR)
frames = divide_into_frames(emphasized_audio, frame_size, hop_size)
plot_audio(SR, frames[0])

# Windowing
windowed_audio = np.vstack([apply_window(window) for window in frames])
plot_audio(SR, windowed_audio[0])

# Zero Padding
target_length = 512
padded_audio = zero_padding(windowed_audio, target_length)
plot_audio(SR, padded_audio[0])

# DFT and truncate
truncated_power_spectrum = calculate_dft(padded_audio)
plot_power(int(SR/2),truncated_power_spectrum[0])

# # Step 8: Mel Filter Banks
# n_mels = 40  # Choose an appropriate value
# mel_frequencies = mel_filter_bank(sr, target_length, n_mels)
#
# # Step 9: Log Spectra
# log_spectra = calculate_log_spectra(magnitude_spectrum)
#
# # Step 10: Cepstral Coefficients
# cepstra = calculate_cepstra(log_spectra)
