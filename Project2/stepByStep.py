import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

IDCT_PADDED_LENGTH = 64

DEFAULT_NUM_FEATURES = 13

HIGH_FREQ = 6855.4976

LOW_FREQ = 133.33

FFT_TARGET_LENGTH = 512

NUM_FILTERS = 40

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


def mel_filter_bank(num_filters, fft_size, sample_rate, low_freq, high_freq):
    # Convert frequencies to Mel scale
    low_mel = 2595 * np.log10(1 + low_freq / 700)
    high_mel = 2595 * np.log10(1 + high_freq / 700)

    # Equally spaced points in Mel scale
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)

    # Convert Mel scale back to Hz
    hz_points = 700 * (10**(mel_points / 2595) - 1)

    # Convert Hz points to bin indices
    bin_indices = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)

    # Create filter banks
    filter_banks = np.zeros((num_filters, int(fft_size / 2 + 1)))

    for i in range(1, num_filters + 1):
        filter_banks[i - 1, bin_indices[i - 1]:bin_indices[i]] = \
            (np.arange(bin_indices[i - 1], bin_indices[i]) - bin_indices[i - 1]) / (bin_indices[i] - bin_indices[i - 1])
        filter_banks[i - 1, bin_indices[i]:bin_indices[i + 1]] = \
            1 - (np.arange(bin_indices[i], bin_indices[i + 1]) - bin_indices[i]) / (bin_indices[i + 1] - bin_indices[i])

    return filter_banks


# Log Spectra
def calculate_log_spectra(magnitude_spectrum):
    return np.log10(magnitude_spectrum)


# DCT
def calculate_cepstra(log_spectra, num_cepstral_coefficients=DEFAULT_NUM_FEATURES):
    return scipy.fftpack.dct(log_spectra, axis=1)[:, :num_cepstral_coefficients]

# IDCT
def inverse_discrete_cosine_transform(cepstra, padded_length):
    cepstra_padded = zero_padding(cepstra, padded_length)
    return scipy.fftpack.idct(cepstra_padded, type=2, axis=1)


# Plot audio wave form
def plot_audio(sr, data):
    length = data.shape[0] / sr
    time = np.linspace(0., length, data.shape[0])
    plt.plot(time, data)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()


# Plot power frequency graphs
def plot_power(max_frequency, data, min_frequency=0):
    frequency = np.linspace(min_frequency, max_frequency, data.shape[0])
    plt.plot(frequency, data)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power")
    plt.show()

# Plot spectrogram
def plot_spectrogram(spectrogram, name):
    plt.matshow(spectrogram.T)
    plt.xlabel('Frame')
    plt.ylabel('Feature')
    plt.title(name)
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
target_length = FFT_TARGET_LENGTH
padded_audio = zero_padding(windowed_audio, target_length)
plot_audio(SR, padded_audio[0])

# DFT and truncate
truncated_power_spectrum = calculate_dft(padded_audio)
plot_power(int(SR/2), truncated_power_spectrum[0])

# Mel filtering
banks = mel_filter_bank(NUM_FILTERS, FFT_TARGET_LENGTH, SR, LOW_FREQ, HIGH_FREQ)
mel_spectra = np.dot(truncated_power_spectrum, banks.T)
plt.plot(mel_spectra[0])
plt.show()


# Log Spectra and plot
mel_log_spectra = calculate_log_spectra(mel_spectra)
plot_spectrogram(mel_log_spectra, "Mel Log Spectrogram")

# ceptra and IDCT derived logspectrum
print(calculate_cepstra(mel_log_spectra).shape)
IDCT_log_spectrogram = inverse_discrete_cosine_transform(calculate_cepstra(mel_log_spectra), IDCT_PADDED_LENGTH)
plot_spectrogram(IDCT_log_spectrogram, "IDCT-derived Log Spectrogram")
