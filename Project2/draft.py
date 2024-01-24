import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


# Read the .wav File
def read_wav(file_path):
    audio, sr = scipy.io.wavfile.read(file_path)
    return audio, sr


# Pre-emphasis
def preemphasis(signal, coefficient=0.97):
    emphasized_signal = np.append(signal[0], signal[1:] - coefficient * signal[:-1])
    return emphasized_signal


# Windowing
def apply_window(signal, window_type='hamming'):
    window = scipy.signal.get_window(window_type, len(signal))
    windowed_signal = signal * window
    return windowed_signal


# Divide into 20ms Windows
def divide_into_windows(signal, window_size, hop_size):
    windows = []
    for i in range(0, len(signal) - window_size + 1, hop_size):
        window = signal[i:i + window_size]
        windows.append(window)
    return np.array(windows)


# Zero Padding
def zero_padding(signal, target_length):
    padded_signal = np.pad(signal, (0, target_length - len(signal)))
    return padded_signal


# Discrete Fourier Transform (DFT)
def calculate_dft(signal):
    return np.fft.fft(signal)


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


# Main Code
FILE_PATH = 'zero.wav'

# Step 3: Read the .wav File
audio, sr = read_wav(FILE_PATH)

# Step 4: Pre-emphasis
emphasized_audio = preemphasis(audio)

# Step 5: Windowing
window_size = int(0.02 * sr)  # 20ms window size
hop_size = int(0.01 * sr)  # 10ms hop size
windows = divide_into_windows(emphasized_audio, window_size, hop_size)
windowed_audio = np.vstack([apply_window(window) for window in windows])

# Step 6: Zero Padding
target_length = 1024  # Choose an appropriate value
padded_audio = zero_padding(windowed_audio, target_length)

# Step 7: Discrete Fourier Transform (DFT)
magnitude_spectrum = np.abs(calculate_dft(padded_audio))

# Step 8: Mel Filter Banks
n_mels = 40  # Choose an appropriate value
mel_frequencies = mel_filter_bank(sr, target_length, n_mels)

# Step 9: Log Spectra
log_spectra = calculate_log_spectra(magnitude_spectrum)

# Step 10: Cepstral Coefficients
cepstra = calculate_cepstra(log_spectra)
