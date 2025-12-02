# 🛡️ MistralGuard Enhanced

MistralGuard Enhanced is a comprehensive anomaly detection and explanation system powered by **Mistral-7B** (via Ollama) and **Explainable AI (XAI)** principles. It combines traditional statistical methods, business rules, and machine learning (Isolation Forest) to detect anomalies in financial or transactional data, and then uses a Large Language Model to explain *why* those anomalies are significant.

## ✨ Features

-   **Multi-Layer Detection**:
    -   **Statistical**: Z-Score analysis for outlier detection.
    -   **Business Rules**: Checks for negative amounts, extreme values, etc.
    -   **Data Quality**: Identifies missing or suspicious categories.
    -   **Machine Learning**: Uses **Isolation Forest** for multivariate anomaly detection.
-   **Explainable AI (XAI)**:
    -   Uses **Mistral-7B** to provide structured analysis (Reasoning, Impact, Recommendation).
    -   **Chat with Data**: Ask questions about the detected anomalies in natural language.
    -   Risk Scoring (1-10) for prioritization.
-   **Interactive Dashboard**:
    -   Visualizations using **Plotly** (Risk Distribution, Anomaly Types, **Time Series Analysis**).
    -   Detailed breakdown of each anomaly.
-   **Configuration**:
    -   Adjustable sensitivity thresholds.
    -   Toggle ML components.

## 🚀 Installation

1.  **Prerequisites**:
    -   Python 3.8+
    -   [Ollama](https://ollama.ai/) installed and running.

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Ollama**:
    Pull the Mistral model:
    ```bash
    ollama pull mistral:7b-instruct
    ```
    Start the Ollama server:
    ```bash
    ollama serve
    ```

## 🏃 Usage

### Enhanced Version (Web UI)
Run the full-featured application with dashboard and chat:
```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860`.

### Basic Version (CLI)
Run the lightweight command-line version:
```bash
python mistralguard.py
```

## 📂 Project Structure

-   `app.py`: **Enhanced Version** (Web UI). Main Gradio application entry point.
-   `src/`:
    -   `detector.py`: Core logic for anomaly detection (Statistical, Rules, ML).
    -   `explainer.py`: Interface with Ollama for generating explanations.
    -   `utils.py`: Helper functions for reporting and file handling.
-   `mistralguard.py`: **Basic Version** (CLI). Simple command-line interface for quick checks.

## 🤖 How it Works

1.  **Upload**: User uploads a CSV file (e.g., transaction logs).
2.  **Detect**: The system scans for anomalies using configured methods.
3.  **Explain**: Detected anomalies are sent to the local LLM to generate a human-readable explanation and risk assessment.
4.  **Visualize**: Results are presented in an interactive dashboard.

## 📊 Visualizations

-   **Risk Distribution**: Shows the distribution of risk scores for anomalies.
-   **Anomaly Types**: Breakdown of different types of anomalies detected.
-   **Time Series Analysis**: Visualizes anomalies over time.

## Author 
Aryan 