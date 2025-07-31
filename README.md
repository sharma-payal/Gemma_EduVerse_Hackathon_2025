# Gemma_EduVerse_Hackathon_2025
🌟 Introduction & Impact
EduVerse is more than just a software project; it's a mission to democratize education. It is an offline, on-device AI assistant that shatters educational barriers, transforming any laptop into a personal tutor for the vast majority. Harnessing the power of Gemma 3n, EduVerse provides accessible, engaging, and powerful learning for everyone—from the visually and hearing-impaired to those with a thirst for knowledge but no internet. This project is dedicated to nurturing curiosity and fuelling the imagination and dreams of people in regions with no internet, proving that the will to learn is the only prerequisite for education.

#Key Features
* Completely Offline: Built to be fully autonomous, EduVerse operates without any internet connection. This is a game-changer for learners in rural or underserved areas, ensuring education is always available.

* Multimodal Interaction: We leveraged Gemma 3n's unique capabilities to enable learning beyond text. Users can upload images of diagrams, math problems, or complex charts and receive detailed explanations.

* Accessible Voice-to-Speech: To support visually-impaired students and facilitate a hands-free learning experience, EduVerse includes a robust text-to-speech engine that reads out the AI's responses.

* Intuitive Voice-to-Text: Learners can speak their questions and prompts directly to the AI, making interaction natural and highly accessible for those who prefer verbal communication.

* Persistent Chat History: Learning is a journey, not a single session. EduVerse allows users to save and load their chat history, ensuring that no progress or context is ever lost.

#Technologies Used
* Ollama with Gemma 3n: This is the heart of our project. We chose Ollama to run the Gemma 3n model locally, allowing us to deliver a fast, powerful, and truly offline AI experience while leveraging its state-of-the-art multimodal abilities.

* Streamlit: We built a clean, intuitive, and user-friendly web-based UI with Streamlit, making the complex underlying technology accessible to learners of all ages.

* Vosk: An offline speech recognition engine that ensures our voice-to-text functionality is private and works without an internet connection.

* pyttsx3: An offline text-to-speech library that brings the AI's responses to life, making the project inclusive for a wide range of users.

* Python Libraries: We utilized Pillow for image processing and sounddevice with NumPy for robust audio handling.

#Installation & Setup
To run EduVerse, please follow these steps. It’s designed to be a simple, one-time setup.

* Install Ollama and Pull the Gemma 3n Model:
First, download and install the Ollama application from ollama.com.
Next, open your terminal and pull the required model:
ollama pull gemma3:4b-it-qat

* Clone the Repository:
Clone this public repository to your local machine:
git clone https://github.com/sharma-payal/Gemma_EduVerse_Hackathon_2025.git
cd Gemma_EduVerse_Hackathon_2025

* Set up the Python Environment:
Install the required Python libraries from the requirements.txt file:
pip install -r requirements.txt

* Download and Extract the Vosk Model:
Download the small Vosk model (vosk-model-small-en-us-0.15) from the Vosk website.
Extract the downloaded folder and place it in your project's root directory, renaming it to vosk_model_small.

* Run the Application:
Launch the Streamlit app from your terminal
streamlit run app.py

# How to Use
* Standard Chat: Type your question in the box and press Enter.

* Voice Interaction: Click "🎤 Speak Your Question," ask your question, and the AI will respond in both text and speech.

* Multimodal Learning: Click "🖼️ Upload an Image," select an image (e.g., a diagram or a math problem), then type your question about it and press Enter.

# Future Vision
While EduVerse is currently optimized for laptops to ensure high performance and a reliable offline experience, this project is just the beginning. Our long-term vision is to scale this technology to a lightweight mobile application for powerful smartphones, and eventually to other portable devices. This would realize the dream of a truly ubiquitous AI teaching assistant, continuing our mission to make education accessible to everyone, everywhere.
