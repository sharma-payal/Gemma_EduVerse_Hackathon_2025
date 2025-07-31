
import streamlit as st
import ollama
import pyttsx3
import os
import vosk
import sounddevice as sd
import numpy as np
import threading
import json
from datetime import datetime
import base64
from PIL import Image
from io import BytesIO

# --- Configuration ---
OLLAMA_MODEL = 'gemma3:4b-it-qat'
VOSK_MODEL_PATH = "vosk_model_small"
SAMPLE_RATE = 16000
CHAT_HISTORY_DIR = "chat_histories"

# --- Ensure chat history directory exists ---
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# --- Initialize Text-to-Speech Engine and ALL Session State Variables EARLY ---
if 'tts_engine' not in st.session_state:
    st.session_state.tts_engine = pyttsx3.init()

if 'speech_rate' not in st.session_state:
    st.session_state.speech_rate = 170
st.session_state.tts_engine.setProperty('rate', st.session_state.speech_rate)

if 'speaking_active' not in st.session_state:
    st.session_state.speaking_active = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_saved_chat" not in st.session_state:
    st.session_state.last_saved_chat = None

if "current_chat_file" not in st.session_state:
    st.session_state.current_chat_file = None

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# --- Functions ---
def speak_text_threaded(text):
    st.session_state.speaking_active = True
    st.session_state.tts_engine.setProperty('rate', st.session_state.speech_rate)
    st.session_state.tts_engine.say(text)
    st.session_state.tts_engine.runAndWait()
    st.session_state.speaking_active = False
    st.rerun()

def stop_speaking():
    if st.session_state.speaking_active:
        st.session_state.tts_engine.stop()
        st.session_state.speaking_active = False
        st.info("Speaking stopped.")
        st.rerun()

@st.cache_resource
def load_vosk_model():
    if not os.path.exists(VOSK_MODEL_PATH):
        st.warning(f"Downloading Vosk small model (approx. 50 MB) to '{VOSK_MODEL_PATH}'. This may take a moment...")
        os.makedirs(VOSK_MODEL_PATH, exist_ok=True)
        st.error(f"Vosk model not found. Please download a small model (e.g., 'vosk-model-small-en-us-0.15') from https://alphacephei.com/vosk/models and extract it into the '{VOSK_MODEL_PATH}' folder.")
        st.stop()
        return None

    try:
        vosk.SetLogLevel(-1)
        return vosk.Model(VOSK_MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading Vosk model: {e}")
        st.error(f"Ensure the Vosk model files are correctly extracted into the '{VOSK_MODEL_PATH}' directory.")
        st.stop()
        return None

vosk_model = load_vosk_model()

# --- Chat History Functions ---
def get_chat_files():
    files = [f for f in os.listdir(CHAT_HISTORY_DIR) if f.endswith(".json")]
    return sorted(files, reverse=True)

def save_chat(chat_name):
    if not chat_name.strip():
        chat_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_Chat")
    file_path = os.path.join(CHAT_HISTORY_DIR, f"{chat_name}.json")
    with open(file_path, "w") as f:
        json.dump(st.session_state.messages, f, indent=4)
    st.success(f"Chat saved as '{chat_name}.json'")
    st.session_state.last_saved_chat = chat_name
    st.session_state.current_chat_file = f"{chat_name}.json"
    st.rerun()

def load_chat(file_name):
    file_path = os.path.join(CHAT_HISTORY_DIR, file_name)
    try:
        with open(file_path, "r") as f:
            st.session_state.messages = json.load(f)
        st.session_state.last_saved_chat = file_name.replace('.json', '')
        st.session_state.current_chat_file = file_name
        st.info(f"Loaded chat from '{file_name}'")
    except Exception as e:
        st.error(f"Error loading chat from '{file_name}': {e}")

def new_chat():
    st.session_state.messages = []
    st.session_state.last_saved_chat = None
    st.session_state.current_chat_file = None
    st.success("Started a new chat session.")
    st.rerun()

# --- Helper function to display image in chat message ---
def display_image_from_base64(base64_string):
    image_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(image_data))
    st.image(img, use_column_width=True)

# --- Streamlit Sidebar ---
with st.sidebar:
    st.title("⚙️ EduVerse Settings")
    st.markdown("---")

    st.subheader("🗣️ Voice Settings")
    new_speech_rate = st.slider(
        "Speech Rate (WPM)",
        min_value=100,
        max_value=300,
        value=st.session_state.speech_rate,
        step=10,
        help="Adjust the words per minute for the AI's voice."
    )
    if new_speech_rate != st.session_state.speech_rate:
        st.session_state.speech_rate = new_speech_rate
        st.session_state.tts_engine.setProperty('rate', st.session_state.speech_rate)

    st.markdown("---")
    st.subheader("💾 Chat History")
    if st.button("✨ Start New Chat", use_container_width=True):
        new_chat()

    chat_name_input = st.text_input(
        "Save Chat As:",
        value=st.session_state.get('last_saved_chat', ''),
        help="Enter a name for your chat session. Leave blank to auto-name."
    )
    if st.button("⬇️ Save Current Chat", use_container_width=True, disabled=not st.session_state.messages):
        save_chat(chat_name_input)

    chat_files = get_chat_files()
    if chat_files:
        selected_chat = st.selectbox(
            "⬆️ Load Saved Chat:",
            options=[""] + chat_files,
            index=0,
            key="load_chat_selectbox",
            help="Select a previously saved chat session to load."
        )
        if st.session_state.load_chat_selectbox and st.session_state.load_chat_selectbox != "":
            load_chat(st.session_state.load_chat_selectbox)
    else:
        st.info("No saved chats found.")


    st.markdown("---")
    st.subheader("💡 About EduVerse")
    st.info("""
        **Technology:** EduVerse is powered by **Gemma 3n**, a state-of-the-art
        open-source AI model from Google. It excels in **multimodal capabilities**
        (processing text and speech, with potential for image/video inputs),
        running entirely **offline** on your device via **Ollama**.
        This ensures unparalleled privacy, security, and accessibility
        without internet dependency.

        **Purpose & Real-World Impact:** EduVerse is designed as a personalized,
        secure, and universally accessible AI learning assistant. It aims to empower:
        * **Students worldwide:** Including those in rural areas or with limited internet,
            providing continuous learning opportunities.
        * **Aspiring individuals:** Offering guidance and knowledge to those dreaming
            of becoming future leaders and innovators, regardless of their current access.
        * **Lifelong learners:** Supporting curious minds of all ages, including
            the elderly, who seek to expand their knowledge.
        * **Individuals in need:** Providing a discreet and accessible source of
            information and advice.
        * **Visually impaired students:** Utilizing its text-to-speech
            capabilities to make education accessible and inclusive.
    """)
    st.markdown("---")


# --- Streamlit Main App ---
st.title("Gemma EduVerse")
st.subheader("Your AI Teaching Assistant for All Ages, Accessible Anywhere")

with st.expander("💡 How to Use Your EduVerse Assistant"):
    st.write("""
    1.  **Ask a Question:** You can either type your question in the box below or use the "🎤 Speak Your Question" button.
    2.  **Upload an Image:** Use the image uploader to ask a question about an image.
    3.  **Listen to the Answer:** The AI will respond in text and also read it aloud.
    4.  **Stop Speaking:** Click "🚫 Stop Speaking" at any time to halt the voice output.
    5.  **No Internet Needed:** After initial setup, your AI works completely offline!
    6.  **Manage Chats:** Use the sidebar to start new chats, save current ones, or load previous conversations.
    """)

st.markdown("---")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if 'image' in message:
            display_image_from_base64(message['image'])
        st.markdown(message["content"])

# --- Image Upload Section ---
st.session_state.uploaded_image = st.file_uploader(
    "🖼️ Upload an Image to ask a question about it:",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    key="image_uploader"
)
if st.session_state.uploaded_image:
    with st.chat_message("user"):
        st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_column_width=True)

# --- Voice Input and Stop Speaking Buttons ---
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🎤 Speak Your Question", key="speak_button", use_container_width=True):
        st.session_state.speaking = True
        st.info("👂 **Listening...** Speak your question now. Press 'Stop Listening' to finish.")
        
        rec = vosk.KaldiRecognizer(vosk_model, SAMPLE_RATE)
        audio_chunks = []
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
                while st.session_state.speaking:
                    data, overflowed = stream.read(SAMPLE_RATE // 4)
                    if overflowed:
                        st.warning("Audio buffer overflowed!")
                    
                    audio_chunks.append(data.copy())

                    if rec.AcceptWaveform(data.tobytes()):
                        result = rec.Result()
                        spoken_text = eval(result)['text']
                        if spoken_text:
                            st.session_state.spoken_input = spoken_text
                            st.session_state.speaking = False
                            break
                    else:
                        partial_result = rec.PartialResult()
                        if 'partial' in eval(partial_result) and eval(partial_result)['partial']:
                            st.session_state.partial_text = eval(partial_result)['partial']
                            
        except Exception as e:
            st.error(f"❌ Error during voice input: {e}. Ensure your microphone is connected and accessible.")
            st.session_state.speaking = False

with col2:
    if st.button("🚫 Stop Speaking", key="stop_speaking_button", disabled=not st.session_state.speaking_active, use_container_width=True):
        stop_speaking()

# --- Main chat input logic ---
if prompt := st.chat_input("Type your question here:", key="text_input"):
    user_message = {"role": "user", "content": prompt}

    if st.session_state.uploaded_image:
        image_bytes = st.session_state.uploaded_image.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        user_message['image'] = base64_image
        
        messages_for_ollama = st.session_state.messages + [
            {"role": "user", "content": prompt, "images": [base64_image]}
        ]
    else:
        messages_for_ollama = st.session_state.messages + [user_message]
    
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        if 'image' in user_message:
            display_image_from_base64(user_message['image'])
        st.markdown(user_message['content'])

    with st.spinner("🧠 Thinking..."):
        try:
            response = ollama.chat(model=OLLAMA_MODEL, messages=messages_for_ollama)
            assistant_response = response['message']['content']
        except Exception as e:
            assistant_response = f"❌ Error from AI: {e}. Please ensure Ollama server is running, '{OLLAMA_MODEL}' is pulled, and it supports multimodal input."

    with st.chat_message("assistant"):
        st.markdown(f"🤖 {assistant_response}")
        st.session_state.speaking_active = True
        speak_thread = threading.Thread(target=speak_text_threaded, args=(assistant_response,))
        speak_thread.start()
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    # Correct way to clear the uploaded image state
    st.session_state.uploaded_image = None