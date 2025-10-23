# Sales-Agent
Agentic AI System for performing sales operations
## 📘 `README.md`

```markdown
# 🎙️ FastAPI + Streamlit AI Voice Assistant

An end-to-end **Generative AI app** built using **FastAPI (backend)** and **Streamlit (frontend)** for real-time speech-to-text, LLM-based reasoning, and text-to-speech response generation.

This project demonstrates:
- 🧠 **FastAPI backend** with Azure / Mistral LLM inference  
- 🎨 **Streamlit frontend** with live microphone input  
- 🔊 **Speech-to-Text (STT)** via `faster-whisper`  
- 🗣️ **Text-to-Speech (TTS)** via `kokoro`  
- ⚡ Real-time API communication between Streamlit and FastAPI using `requests`

---

## 🚀 Features
✅ FastAPI backend serving AI endpoints  
✅ Streamlit frontend for user interaction  
✅ Audio recording and playback  
✅ Real-time LLM responses  
✅ Ready for Azure or local GPU deployment  

---

## 🧩 Project Structure
```

project_folder/

- booking_backend_fastapi.py     # FastAPI backend
- booking_frontend_fast.py       # Streamlit frontend
- requirements.txt               # Dependencies
- env/                           # Virtual environment

````

---

## ⚙️ Installation

### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
````

### 2️⃣ Create and activate virtual environment

```bash
python -m venv env
# Windows
.\env\Scripts\activate
# macOS/Linux
source env/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

> 💡 If you face build issues on Windows (e.g., *Microsoft C++ Build Tools required*),
> install [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

---

## 🖥️ Running the Application

### ▶️ Step 1: Start FastAPI backend

In **Terminal 1**:

```bash
uvicorn booking_backend_fastapi:app --reload --port 8000
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

---

### ▶️ Step 2: Start Streamlit frontend

In **Terminal 2**:

```bash
streamlit run booking_frontend_fast.py
```

✅ Streamlit will automatically open in your browser:
[http://localhost:8501](http://localhost:8501)

---

## 🔗 Backend–Frontend Connection

Make sure your Streamlit app points to:

```python
BACKEND_URL = "http://127.0.0.1:8000"
```

You can modify this to use your Azure API endpoint or public URL when deploying.

---

## 🧠 Example Workflow

1. 🎙️ Record or upload your voice in Streamlit
2. 🪄 The audio is sent to the FastAPI backend
3. 🧾 Backend performs:

   * Speech-to-Text (via faster-whisper)
   * Query processing (via Mistral/Azure LLM)
   * Text-to-Speech (via Kokoro)
4. 🔊 The response audio is sent back and played in Streamlit

---

## 🖼️ Screenshots

### 🧩 Streamlit Interface

![Streamlit UI](assets/streamlit_ui.png)

### ⚙️ FastAPI Server Running

![FastAPI Terminal](assets/fastapi_terminal.png)

### 🧠 Response Example

![Response Output](assets/response_output.png)

*(You can add your own screenshots to `/assets` folder for visual appeal.)*

---

## ☁️ Deployment Options

### 🔹 Streamlit Cloud

1. Push your repo to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. Deploy the app using your GitHub repo

### 🔹 Azure App Service

Use this command in Azure CLI:

```bash
az webapp up --name your-app-name --resource-group your-group --sku B1
```

Ensure your `requirements.txt` includes:

```
fastapi
uvicorn
gunicorn
```

---

## 🧰 Tech Stack

| Component   | Library/Tool           |
| ----------- | ---------------------- |
| Frontend    | Streamlit              |
| Backend     | FastAPI + Uvicorn      |
| STT         | faster-whisper         |
| TTS         | kokoro                 |
| LLM         | Mistral / Azure OpenAI |
| Audio       | soundfile, playsound   |
| Environment | Python 3.10+           |

---

## 🧪 Sample API Test (Backend)

Use `curl` or `Postman`:

```bash
curl -X POST http://127.0.0.1:8000/process_audio \
  -F "file=@sample.wav"
```

Response:

```json
{
  "transcription": "Hello, how can I help you?",
  "response": "I can assist with your query.",
  "audio_url": "/output/response.wav"
}
```

---

## 👨‍💻 Author

**Jagadeesh Pamanji**
*Generative AI Engineer*
🔗 [LinkedIn](https://linkedin.com/in/your-profile)
📧 [your.email@example.com](mailto:your.email@example.com)

---

## ⭐ Contribute

Pull requests are welcome!
If you find this project helpful, please give it a ⭐ on GitHub!

---

## 🛡️ License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

```

---

## 🪄 Next Steps (optional)
To make this README visually rich:
- Add 3 PNGs in a folder named `/assets`  
  - `streamlit_ui.png`
  - `fastapi_terminal.png`
  - `response_output.png`
- Update image links in README to point to your GitHub repo.

---

Would you like me to **generate those sample screenshots** (`streamlit_ui.png`, `fastapi_terminal.png`, and `response_output.png`) for you so you can directly upload them to GitHub?
```
