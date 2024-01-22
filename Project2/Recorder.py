import wave
import pyaudio
import numpy as np
import keyboard

BACKGROUND_CALCULATION_FRAMES = 10

# Recording parameters
FRAME_SIZE = 1024
SAMPLE_FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000  # or 44100 for Mac users
FILENAME = "output.wav"

# Endpointing algorithm parameters
FORGET_FACTOR = 1.5
ADJUSTMENT = 0.05
THRESHOLD = 2

# Calculate the decibel of a frame of data points
def energy_per_sample_in_decibel(frame):
    energy = np.sum(np.frombuffer(frame, dtype=np.int16).astype(float) ** 2) / (len(frame) / 2)
    return 10 * np.log10(energy)


# The adaptive endpointing algorithm
def classify_frame(audioframe, background, level, forgetfactor, threshold, adjustment):
    current = energy_per_sample_in_decibel(audioframe)
    isSpeech = False
    level = ((level * forgetfactor) + current) / (forgetfactor + 1)

    if current < background:
        background = current
    else:
        background += (current - background) * adjustment

    if level < background:
        level = background
    if (level - background > threshold):
        isSpeech = True

    return isSpeech, background, level


# Initialize the PortAudio interface
p = pyaudio.PyAudio()

print('Press "space" to start recording')

# Wait for the 'space' key to be pressed
keyboard.wait('space')

print('Recording')

# Open the stream for recording
stream = p.open(format=SAMPLE_FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=FRAME_SIZE)

frames = []
current_background = 0
current_level = 0
current_isSpeech = False

# Record until silence is detected
while True:
    audioframe = stream.read(FRAME_SIZE)
    frames.append(audioframe)
    # Record at least 10 frames
    if len(frames) < BACKGROUND_CALCULATION_FRAMES:
        continue
    elif len(frames) == BACKGROUND_CALCULATION_FRAMES:
        # Initialize background and level once the first 10 frames are recorded
        current_level = energy_per_sample_in_decibel(audioframe)
        current_background = energy_per_sample_in_decibel(b''.join(frames))
    else:
        current_isSpeech, current_background, current_level = classify_frame(audioframe, current_background,
                                                                             current_level, FORGET_FACTOR, THRESHOLD,
                                                                             ADJUSTMENT)
        if not current_isSpeech:
            break

# Stop and close the stream
stream.stop_stream()
stream.close()

# Terminate the PortAudio interface
p.terminate()

print('Finished recording')

# Save the recorded data as a WAV file
wf = wave.open(FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(SAMPLE_FORMAT))
wf.setframerate(SAMPLE_RATE)
wf.writeframes(b''.join(frames))
wf.close()
