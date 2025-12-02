import requests
import logging
import json
import numpy as np 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaExplainer:
    """Enhanced Ollama integration for XAI"""
    
    def __init__(self, base_url="http://localhost:11434", model="mistral:7b-instruct"):
        self.base_url = base_url
        self.model = model
        self.is_available = self.check_connection()
    
    def check_connection(self):
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                logger.info(f"✅ Ollama connected. Models: {model_names}")
                return any('mistral' in name.lower() for name in model_names)
            return False
        except:
            logger.warning("⚠️ Ollama not available - using fallback")
            return False

    def explain_anomaly(self, record, issues):
        """Generate explanation using LLM"""
        if not self.is_available:
            return self._fallback_explanation(issues)

        # Construct a detailed prompt for XAI
        issues_desc = "\n".join([f"- {i['type']}: {i.get('issue', 'Unknown Issue')} (Value: {i.get('value', 'N/A')})" for i in issues])
        
        prompt = f"""
        You are an AI Financial Analyst. Analyze this anomalous transaction record and explain WHY it is suspicious.
        
        Record Data:
        {json.dumps(record, default=str, indent=2)}
        
        Detected Issues:
        {issues_desc}
        
        Provide a structured analysis in JSON format with these keys:
        1. "reasoning": A clear explanation of why this is an anomaly.
        2. "impact": Potential business impact (e.g., Fraud risk, Data Error).
        3. "recommendation": Actionable advice (e.g., Verify with merchant, Correct data).
        4. "risk_score": A score from 1-10 based on severity.
        
        Keep the explanation concise but professional.
        """

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json", # Request JSON output if supported by model version, otherwise prompt implies it
                    "options": {"temperature": 0.2, "num_predict": 200}
                },
                timeout=300
            )
            
            if response.status_code == 200:
                llm_response = response.json().get('response', '')
                try:
                    # Try to parse JSON
                    analysis = json.loads(llm_response)
                    return analysis
                except json.JSONDecodeError:
                    # Fallback if model didn't return valid JSON
                    return {
                        "reasoning": llm_response,
                        "impact": "Unstructured analysis",
                        "recommendation": "Review manually",
                        "risk_score": 5
                    }
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            
        return self._fallback_explanation(issues)

    def chat_with_data(self, query, report_context):
        """Chat with the analyzed data context"""
        if not self.is_available:
            return "⚠️ AI is not available. Please ensure Ollama is running."

        # Create a context summary from the report
        # Limit context to avoid token limits
        context_str = json.dumps(report_context[:10], default=str) # Top 10 anomalies as context
        
        prompt = f"""
        You are an AI Analyst assisting a user with an anomaly detection report.
        
        Context (Top Anomalies detected):
        {context_str}
        
        User Question: {query}
        
        Answer the user's question based on the provided context. Be helpful and concise.
        If the answer isn't in the context, say so.
        """
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 300}
                },
                timeout=90
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
        except Exception as e:
            return f"Error generating response: {str(e)}"
            
        return "I couldn't generate a response at this time."

    def _fallback_explanation(self, issues):
        """Rule-based fallback explanation"""
        types = [i['type'] for i in issues]
        severity = "High" if any(i.get('severity') == 'High' for i in issues) else "Medium"
        
        return {
            "reasoning": f"Detected anomalies: {', '.join(types)}. Statistical deviation or rule violation found.",
            "impact": "Potential data quality issue or outlier.",
            "recommendation": "Manual verification required.",
            "risk_score": 8 if severity == "High" else 5
        }
