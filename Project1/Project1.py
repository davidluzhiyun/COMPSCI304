import wave
import pyaudio
import numpy as np
import keyboard

def energy_per_sample_in_decibel(frame):
    energy = np.sum(np.frombuffer(frame, dtype=np.int16).astype(float)**2) / len(frame)
    return 10 * np.log10(energy)

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

# Recording parameters
chunk = 1024
sample_format = pyaudio.paInt16
channels = 1
fs = 44100  # or 44100 for Mac users
filename = "output.wav"

# Endpointing algorithm parameters
forgetfactor = 1.5
adjustment = 0.025
threshold = 15

# Initialize the PortAudio interface
p = pyaudio.PyAudio()

print('Press "space" to start recording')

# Wait for the 'space' key to be pressed
keyboard.wait('space')

print('Recording')

# Open the stream for recording
stream = p.open(format=sample_format, channels=channels, rate=fs, input=True, frames_per_buffer=chunk)

frames = []
background = 0
level = 0
isSpeech = False

# Record until silence is detected
while True:
    audioframe = stream.read(chunk)
    isSpeech, background, level = classify_frame(audioframe, background, level, forgetfactor, threshold, adjustment)
    frames.append(audioframe)
    if not isSpeech and len(frames) > 10:  # Ensure some audio has been recorded before stopping
        break

# Stop and close the stream
stream.stop_stream()
stream.close()

# Terminate the PortAudio interface
p.terminate()

print('Finished recording')

# Save the recorded data as a WAV file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()
