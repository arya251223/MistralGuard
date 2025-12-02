# 🛡️ MistralGuard

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-5.0%2B-orange?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![Ollama](https://img.shields.io/badge/Ollama-Mistral%207B-black?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.ai/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**MistralGuard** is a next-generation anomaly detection and explanation system that bridges the gap between traditional statistical monitoring and modern Generative AI. 

Powered by **Mistral-7B** (via Ollama) and **Explainable AI (XAI)** principles, it not only detects *when* something goes wrong but explains *why*—in plain English.

🔗 **GitHub Repository**: [https://github.com/arya251223/MistralGuard](https://github.com/arya251223/MistralGuard)

---

## ✨ Key Features

### 🧠 Intelligent Detection
MistralGuard employs a multi-layered approach to ensure no anomaly goes unnoticed:
-   **Statistical Analysis**: Real-time Z-Score calculation for outlier detection.
-   **Business Logic**: Configurable rules to catch domain-specific errors (e.g., negative transactions, extreme values).
-   **Data Quality Checks**: Identifies missing data, nulls, or suspicious categories.
-   **Machine Learning**: Integrated **Isolation Forest** algorithm for detecting complex, multivariate anomalies.

### 🗣️ Explainable AI (XAI)
Don't just get an alert; get an explanation.
-   **Natural Language Insights**: Uses Mistral-7B to generate structured reports (Reasoning, Impact, Recommendation).
-   **Chat with Data**: Interactive chat interface to ask follow-up questions about specific anomalies.
-   **Risk Scoring**: AI-assigned risk scores (1-10) to help prioritize your response.

### 📊 Interactive Dashboard
Built with **Gradio** and **Plotly** for a seamless user experience:
-   **Real-time Visualizations**: Risk distribution histograms, anomaly type pie charts, and interactive time-series plots.
-   **Data View**: Inspect the raw data and identified anomalies directly in the browser.
-   **Exportable Reports**: Download comprehensive CSV reports with AI insights included.

---

## 🛠️ Tech Stack

-   **Core**: Python 3.12+
-   **AI/LLM**: Ollama (Mistral-7B Instruct)
-   **Frontend**: Gradio (Soft Theme)
-   **Data Processing**: Pandas, NumPy
-   **Machine Learning**: Scikit-learn (Isolation Forest)
-   **Visualization**: Plotly Express

---

## 🚀 Installation

### 1. Prerequisites
-   **Python 3.8+** installed.
-   **[Ollama](https://ollama.ai/)** installed and running.

### 2. Clone the Repository
```bash
git clone https://github.com/arya251223/MistralGuard.git
cd MistralGuard
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup AI Model
Pull the Mistral model (or any other supported model):
```bash
ollama pull mistral:7b-instruct
```
*Ensure the Ollama server is running (`ollama serve`).*

---

## 🏃 Usage

### 🌟 Enhanced Version (Recommended)
Launch the full web application with the interactive dashboard:
```bash
python app.py
```
> Open your browser at `http://127.0.0.1:7860`

### ⚡ Basic Version (CLI)
For quick, headless checks via command line:
```bash
python mistralguard.py
```

---

## 📂 Project Structure

```
MistralGuard/
├── app.py                 # 🌟 Main Web Application (Gradio)
├── mistralguard.py        # ⚡ Basic CLI Version
├── requirements.txt       # Project Dependencies
├── README.md              # Documentation
└── src/
    ├── detector.py        # Anomaly Detection Logic (Stats, ML, Rules)
    ├── explainer.py       # LLM Integration & Prompt Engineering
    └── utils.py           # Reporting & Helper Functions
```

## � Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 👤 Author

**Aryan**

---
*Built with ❤️ using Python and Mistral AI.*
