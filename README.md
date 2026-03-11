# 🤖 Multi-Document RAG Search Engine (with Real-Time Web Search)

An intelligent AI assistant built using **Streamlit and Large Language Models (LLMs)** that can answer questions using uploaded documents or perform web search when required.

This project demonstrates how **Retrieval Augmented Generation (RAG)** can be used to build practical AI systems capable of understanding documents and generating contextual answers.

---

# 🚀 Features

* 📄 Upload and analyze PDF documents
* 🧠 LLM-powered question answering
* 🔎 Retrieval Augmented Generation (RAG)
* 🌐 Optional Web Search mode
* ⚡ Interactive interface using Streamlit

---

# 🛠 Tech Stack

* Python
* Streamlit
* Google Gemini API
* PyPDF
* Tavily Web Search API
* Python Dotenv

---

# 📁 Project Structure

```
AGENTIC-AI-PROJECT/

├── app.py        # Main Streamlit application
├── .env          # API keys and environment variables
├── .gitignore    # Files ignored by Git
└── venv/         # Local virtual environment (not pushed to GitHub)
```

Note: The `venv` folder is used only for local development and should **not** be uploaded to GitHub.

---

# ⚙️ Installation

Clone the repository

```
git clone https://github.com/your-username/agentic-ai-project.git
```

Move into the project folder

```
cd agentic-ai-project
```

---

# 🧪 Create Virtual Environment

```
python -m venv venv
```

Activate the environment

### Windows

```
venv\Scripts\activate
```

### Mac / Linux

```
source venv/bin/activate
```

---

# 📦 Install Dependencies

Install required libraries

```
pip install streamlit google-generativeai pypdf tavily-python python-dotenv
```

(Optional) Create a requirements file

```
pip freeze > requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root and add your API keys.

Example:

```
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

# ▶️ Running the Application

Start the Streamlit server

```
streamlit run app.py
```

After running the command, Streamlit will launch the application automatically.

If it does not open automatically, go to:

```
http://localhost:8501
```

---

# 💡 How the Application Works

1. The user uploads a document
2. The application extracts text from the document
3. The text is processed and sent to the LLM
4. The model generates contextual answers
5. If **Web Mode** is enabled, the system searches the internet and returns relevant results

---

# 👨‍💻 Author

**Harsh**
Aspiring Data Scientist | AI Engineer

---

# ⭐ Future Improvements

* Support for multiple document formats
* Chat history memory
* Improved retrieval pipeline
* Deployment using Docker or Cloud platforms
