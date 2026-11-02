
import ollama
import speech_recognition as sr
import pyttsx3
import pyaudio
import numpy as np
import time

from openwakeword.model import Model


# ----------------------------------
# SPIDY CONFIGURATION
# ----------------------------------

speaker = pyttsx3.init()
speaker.setProperty("rate", 180)
speaker.setProperty("volume", 1.0)

recognizer = sr.Recognizer()

# Load wake-word model
wake_model = Model(
    wakeword_models=["hey_jarvis"],
    inference_framework="onnx"
)

# Microphone configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280


# ----------------------------------
# SPIDY'S PERSONALITY AND MEMORY
# ----------------------------------

messages = [
    {
        "role": "system",
        "content": (
            "Your name is Spidy. You are a clever, friendly "
            "voice assistant inspired by a superhero AI. "
            "Keep answers concise and natural when spoken aloud. "
            "You can answer questions but cannot control the "
            "computer unless a tool is explicitly provided."
        )
    }
]


# ----------------------------------
# TEXT TO SPEECH
# ----------------------------------

def speak(text):
    print(f"\n🕷️ SPIDY: {text}")
    speaker.say(text)
    speaker.runAndWait()


# ----------------------------------
# WAKE WORD DETECTION
# ----------------------------------

def wait_for_wake_word():

    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("\n🕷️ Spidy is waiting for 'Hey Jarvis'...")

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

            prediction = wake_model.predict(audio_frame)

            score = prediction.get("hey_jarvis", 0)

            if score > 0.5 and time.time() - last_detection > 2:

                print("\nWake word detected!")

                last_detection = time.time()
                return

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


# ----------------------------------
# SPEECH RECOGNITION
# ----------------------------------

def listen():

    with sr.Microphone() as source:

        print("\n🎤 Listening for your command...")

        recognizer.adjust_for_ambient_noise(
            source,
            duration=0.5
        )

        try:
            audio = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=12
            )

        except sr.WaitTimeoutError:
            print("No speech detected.")
            return None

    try:
        text = recognizer.recognize_google(audio)

        print(f"YOU: {text}")

        return text

    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
        return None

    except sr.RequestError:
        print("Speech recognition service unavailable.")
        return None


# ----------------------------------
# AI PROCESSING
# ----------------------------------

def think(user_text):

    messages.append({
        "role": "user",
        "content": user_text
    })

    try:

        response = ollama.chat(
            model="qwen2.5:3b",
            messages=messages
        )

        answer = response["message"]["content"]

        messages.append({
            "role": "assistant",
            "content": answer
        })

        if len(messages) > 13:
            del messages[1:3]

        return answer

    except Exception as error:

        print(f"AI error: {error}")

        return "My AI brain is unavailable. Please check Ollama."


# ----------------------------------
# MAIN PROGRAM
# ----------------------------------

def main():

    print("=" * 40)
    print("       PROJECT SPIDY")
    print("=" * 40)

    speak("Spidy is online. Say Hey Jarvis to activate me.")

    while True:

        # Wait until wake word is detected
        wait_for_wake_word()

        # Wake word detected
        speak("I'm listening.")

        # Listen to the actual command
        user_text = listen()

        if not user_text:
            continue

        if user_text.lower().strip() in [
            "exit",
            "quit",
            "stop spidy",
            "goodbye spidy"
        ]:

            speak("Signing off. See you later!")
            break

        # Process command
        answer = think(user_text)

        # Speak response
        speak(answer)


if __name__ == "__main__":
    main()
