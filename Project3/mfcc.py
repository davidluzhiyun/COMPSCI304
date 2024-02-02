import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

import librosa

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
    assert len(signal[0]) <= target_length
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

# Normalization
def normalize_features(cepstra):
    # Mean subtraction
    normalized_cepstra = cepstra - np.mean(cepstra, axis=0, keepdims=True)

    # Variance normalization
    normalized_cepstra /= np.std(normalized_cepstra, axis=0, keepdims=True)

    return normalized_cepstra

# IDCT
def inverse_discrete_cosine_transform(cepstra, padded_length):
    cepstra_padded = zero_padding(cepstra, padded_length)
    return scipy.fftpack.idct(cepstra_padded, type=2, axis=1)

# Taking delta
# assume delta(0) = delta(1) delta(-1) = delta(-2) for boundaries
def delta_features(features):
    # Calculate delta for the interior frames
    delta_interior = features[2:] - features[:-2]

    # Calculate delta for the first frame
    delta_first = features[1] - features[0]

    # Calculate delta for the last frame
    delta_last = features[-1] - features[-2]

    # Combine delta for all frames
    delta = np.vstack([delta_first, delta_interior, delta_last])

    return delta


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

# Extra feature, delta feature and double delta feature (each occupying 1/3 in this order)
# axis 0 for frames axis 1 for features
def extract_feature(filename, frame_time = FRAME_TIME, hop_time = HOP_TIME, window_type = DEFAULT_WINDOW_TYPE, fft_target_length = FFT_TARGET_LENGTH, num_filters = NUM_FILTERS, low_frequency = LOW_FREQ, high_frequency = HIGH_FREQ, num_cepstral_coefficients=DEFAULT_NUM_FEATURES, IDCT_padded_length = IDCT_PADDED_LENGTH):
    sr, data = read_wav(filename)

    # Pre-emphasis
    emphasized_audio = preemphasis(data)

    # To frame
    frame_size = int(frame_time * sr)
    hop_size = int(hop_time * sr)
    frames = divide_into_frames(emphasized_audio, frame_size, hop_size)

    # Windowing
    windowed_audio = np.vstack([apply_window(window, window_type) for window in frames])

    # Zero Padding
    padded_audio = zero_padding(windowed_audio, fft_target_length)

    # DFT and truncate
    truncated_power_spectrum = calculate_dft(padded_audio)

    # Mel filtering
    banks = mel_filter_bank(num_filters, fft_target_length, sr, low_frequency, high_frequency)
    mel_spectra = np.dot(truncated_power_spectrum, banks.T)

    # Log Spectra
    mel_log_spectra = calculate_log_spectra(mel_spectra)

    # cepstra
    cepstra = calculate_cepstra(mel_log_spectra, num_cepstral_coefficients)
    normalized_cepstra = normalize_features(cepstra)
    delta1 = delta_features(normalized_cepstra)
    delta2 = delta_features(delta1)

    # # Tested such that result similar to librosa under these parameters
    # # difference in length comes from  difference between scipy.io.wavfile.read() and librosa.load()
    # # Also result is transposed comparing to each other
    # # For ours: axis 0: frames, axis 1: features
    # cepstra2 = librosa.feature.mfcc(y=librosa.load(filename)[0], sr=librosa.load(filename)[1], n_mfcc=13, norm=None,n_fft=FFT_TARGET_LENGTH, hop_length=frame_size//2, win_length=frame_size, window=DEFAULT_WINDOW_TYPE, center=False, power=10)
    #
    # plot_spectrogram(cepstra, "my")
    # plot_spectrogram(cepstra2.T, "lib")
    #
    # plot_spectrogram(inverse_discrete_cosine_transform(normalize_features(cepstra), padded_length=IDCT_padded_length),"my")
    # plot_spectrogram(inverse_discrete_cosine_transform(normalize_features(cepstra2.T), padded_length=IDCT_padded_length),"lib")

    normalized_cepstra = normalize_features(cepstra)

    # Test normalization
    # plot_spectrogram(inverse_discrete_cosine_transform(cepstra, padded_length=IDCT_padded_length),"my")
    # plot_spectrogram(inverse_discrete_cosine_transform(normalize_features(cepstra), padded_length=IDCT_padded_length),"lib")

    delta1 = delta_features(normalized_cepstra)
    delta2 = delta_features(delta1)

    extracted_features = np.hstack([normalized_cepstra, delta1, delta2])

    return extracted_features
