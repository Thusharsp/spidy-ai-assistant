
import ollama
import speech_recognition as sr
import pyttsx3

# Spidy's voice
speaker = pyttsx3.init()
speaker.setProperty("rate", 180)
speaker.setProperty("volume", 1.0)

# Spidy's microphone
recognizer = sr.Recognizer()

# Spidy's personality and conversation memory
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


def speak(text):
    print(f"\n🕷️ SPIDY: {text}")
    speaker.say(text)
    speaker.runAndWait()


def listen():
    with sr.Microphone() as source:
        print("\n🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)

        try:
            audio = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=12
            )
        except sr.WaitTimeoutError:
            print("No speech detected. Try again.")
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

        # Keep the conversation history bounded.
        if len(messages) > 13:
            del messages[1:3]

        return answer

    except Exception as error:
        print(f"AI error: {error}")
        return "My AI brain is unavailable. Please check Ollama."


def main():
    print("=" * 40)
    print("       PROJECT SPIDY")
    print("=" * 40)

    speak("Spidy is online. How can I help you?")
    speak("SPEAK YOU BLUDDY..!")

    while True:
        user_text = listen()

        if not user_text:
            continue

        if user_text.lower().strip() in [
            "exit", "quit", "stop spidy", "goodbye spidy"
        ]:
            speak("Signing off. See you later!")
            break

        answer = think(user_text)
        speak(answer)


if __name__ == "__main__":
    main()