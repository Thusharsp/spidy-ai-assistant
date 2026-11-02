import pyaudio
import numpy as np
import time

from openwakeword.model import Model

# Load wake-word model
model = Model(
    wakeword_models=["hey_jarvis"],
    inference_framework="onnx"
)

# Microphone configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280  # 80 milliseconds of audio

audio = pyaudio.PyAudio()

stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

print("Spidy is listening for 'Hey Jarvis'...")

last_detection = 0

try:
    while True:
        audio_data = stream.read(
            CHUNK,
            exception_on_overflow=False
        )

        audio_frame = np.frombuffer(
            audio_data,
            dtype=np.int16
        )

        prediction = model.predict(audio_frame)

        score = prediction.get("hey_jarvis", 0)

        if score > 0.5 and time.time() - last_detection > 2:
            print("Wake word detected!")
            last_detection = time.time()

except KeyboardInterrupt:
    print("\nStopping wake-word detection...")

finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()