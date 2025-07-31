
import ollama
import sys
import pyttsx3
from vosk import Model, KaldiRecognizer, SetLogLevel # Corrected: SetLogLevel
import cv2 # Used by opencv-python-headless
from PIL import Image # Pillow library

print("--- Starting Gemma Offline EduVerse Environment Test ---")

# --- Test Ollama/Gemma 3n Interaction ---
print("\nAttempting to connect to Ollama and Gemma 3n...")
try:
    # Test a simple chat interaction
    response = ollama.chat(model='gemma3:4b', messages=[
        {'role': 'user', 'content': 'Explain the concept of photosynthesis in simple terms for a 10-year-old.'}
    ])
    print("\n--- Gemma 3n Response ---")
    print(response['message']['content'])
    print("\nOllama/Gemma 3n test successful!")
except Exception as e:
    print(f"\nERROR: Could not connect to Ollama or run Gemma 3n.")
    print(f"Details: {e}")
    print("Please ensure Ollama desktop application is running and 'ollama pull gemma3:4b' was successful.")
    sys.exit(1) # Exit if core AI component fails

# --- Test pyttsx3 (Text-to-Speech) ---
print("\n--- Basic pyttsx3 (Text-to-Speech) Test ---")
try:
    engine = pyttsx3.init()
    # You can adjust voice properties here if you want
    # voices = engine.getProperty('voices')
    # engine.setProperty('voice', voices[0].id) # Try changing index for different voices
    engine.say("Hello from Gemma Offline EduVerse! Your text to speech is working.")
    engine.runAndWait()
    print("pyttsx3 test successful!")
except Exception as e:
    print(f"ERROR: pyttsx3 test failed.")
    print(f"Details: {e}")
    print("Make sure necessary system TTS components are installed (e.g., espeak-ng on Linux, or system voice drivers on Mac/Windows).")

# --- Test Vosk (Speech-to-Text - Library Import Only) ---
print("\n--- Basic Vosk (Speech-to-Text) Library Import Test ---")
try:
    SetLogLevel(-1) # Corrected: SetLogLevel
    # You don't need to load a model for just checking the import.
    # We'll load a small Vosk model later when we build the voice feature.
    print("Vosk library imported successfully.")
except Exception as e:
    print(f"ERROR: Vosk library import test failed.")
    print(f"Details: {e}")
    print("Ensure 'vosk' is correctly installed in your virtual environment and the function call uses SetLogLevel.")

# --- Test Pillow & OpenCV (Image Processing Libraries) ---
print("\n--- Basic Pillow & OpenCV (Image Processing) Library Import Test ---")
try:
    _ = Image.new('RGB', (1,1)) # Simple Pillow test
    # cv2.cuda.getCudaEnabledDeviceCount() might fail if no CUDA GPU or drivers
    # A safer import test for cv2:
    if cv2.__version__: # Checks if cv2 imported a version string
        print("OpenCV library imported successfully.")
    else:
        raise ImportError("OpenCV version not found, possibly import issue.")
    print("Pillow and OpenCV libraries imported successfully.")
except Exception as e:
    print(f"ERROR: Pillow or OpenCV library import test failed.")
    print(f"Details: {e}")
    print("Ensure 'Pillow' and 'opencv-python-headless' are correctly installed.")


print("\n--- All core environment tests completed. ---")
print("If all tests passed, you are ready to start building the application!")