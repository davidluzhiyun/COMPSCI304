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


# Initialize the PortAudio interface
p = pyaudio.PyAudio()

print('Press "space" to start recording')

# Wait for the 'space' key to be pressed
keyboard.wait('space')

print('Recording')

print('Press "space" to end recording')
# Open the stream for recording
stream = p.open(format=SAMPLE_FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=FRAME_SIZE)


frames = []


# Record
while True:
    data = stream.read(FRAME_SIZE)
    frames.append(data)
    if keyboard.is_pressed('space'):
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
