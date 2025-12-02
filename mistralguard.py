"""
app_with_llm.py - Complete working version with all fixes
Comprehensive Data Validation with AI-Powered Anomaly Detection
"""

import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime
import json
import requests
import logging
import io
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaExplainer:
    """Enhanced Ollama integration with batch processing"""
    
    def __init__(self, base_url="http://localhost:11434", model="mistral:7b-instruct"):
        self.base_url = base_url
        self.model = model
        self.is_available = self.check_connection()
        self.use_fallback = False
    
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
    
    def test_connection(self):
        """Test Ollama with a simple query"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "Reply with OK",
                    "stream": False,
                    "options": {"num_predict": 10}
                },
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def explain_anomaly_batch(self, anomaly_data, use_llm=True):
        """Generate explanation for anomaly with proper categorization"""
        details = anomaly_data.get('anomaly_details', [])
        record = anomaly_data.get('record', {})
        
        # Determine anomaly type and severity
        anomaly_types = []
        risk_level = "Low"
        
        for detail in details:
            if 'deviation' in detail:
                dev = abs(detail['deviation'])
                if dev > 5:
                    anomaly_types.append(f"Extreme outlier ({dev:.1f} std deviations)")
                    risk_level = "High"
                elif dev > 4:
                    anomaly_types.append(f"Severe outlier ({dev:.1f} std deviations)")
                    risk_level = "High"
                elif dev > 3:
                    anomaly_types.append(f"Moderate outlier ({dev:.1f} std deviations)")
                    if risk_level != "High":
                        risk_level = "Medium"
            
            if detail.get('issue') == 'Negative value':
                anomaly_types.append("Negative amount")
                risk_level = "High"
            
            if detail.get('issue') == 'Suspicious category':
                value = detail.get('value', '')
                if value in ['', 'None', 'nan', None]:
                    anomaly_types.append("Missing category")
                    if risk_level == "Low":
                        risk_level = "Medium"
                elif value in ['Unknown', 'UNKNOWN', 'Suspicious']:
                    anomaly_types.append("Suspicious category")
                    risk_level = "Medium"
            
            if detail.get('issue') == 'Missing category':
                anomaly_types.append("Missing category")
                if risk_level == "Low":
                    risk_level = "Low"  # Keep missing categories as low risk
        
        # Generate explanation
        if use_llm and self.is_available and not self.use_fallback:
            try:
                prompt = f"""Brief analysis of anomaly:
Type: {', '.join(anomaly_types)}
Amount: ${record.get('amount', 0):.2f}
Explain in 1 sentence why this needs review."""
                
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 50}
                    },
                    timeout=5
                )
                
                if response.status_code == 200:
                    llm_response = response.json().get('response', '')
                    return {
                        'explanation': llm_response,
                        'risk_level': risk_level,
                        'anomaly_types': anomaly_types
                    }
            except:
                pass
        
        # Fallback explanation
        if anomaly_types:
            explanation = f"Detected: {', '.join(anomaly_types)}. "
            if risk_level == "High":
                explanation += "Immediate review required."
            elif risk_level == "Medium":
                explanation += "Review recommended."
            else:
                explanation += "Monitor for patterns."
        else:
            explanation = "Anomaly detected based on data patterns."
        
        return {
            'explanation': explanation,
            'risk_level': risk_level,
            'anomaly_types': anomaly_types
        }

def detect_anomalies_comprehensive(df):
    """Comprehensive anomaly detection with categorization"""
    anomalies = {}
    
    # 1. Statistical outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        if df[col].nunique() < 3:
            continue
            
        mean = df[col].mean()
        std = df[col].std()
        
        if std > 0:
            for idx in df.index:
                value = df.loc[idx, col]
                deviation = (value - mean) / std
                
                if abs(deviation) > 3:
                    if idx not in anomalies:
                        anomalies[idx] = []
                    anomalies[idx].append({
                        'type': 'statistical',
                        'column': col,
                        'value': float(value),
                        'mean': float(mean),
                        'std': float(std),
                        'deviation': float(deviation)
                    })
    
    # 2. Business rule violations
    if 'amount' in df.columns:
        # Negative amounts
        for idx in df[df['amount'] < 0].index:
            if idx not in anomalies:
                anomalies[idx] = []
            anomalies[idx].append({
                'type': 'business_rule',
                'column': 'amount',
                'value': float(df.loc[idx, 'amount']),
                'issue': 'Negative value'
            })
        
        # Extremely high amounts (>99th percentile)
        if len(df['amount']) > 0:
            threshold_99 = df['amount'].quantile(0.99)
            for idx in df[df['amount'] > threshold_99].index:
                if idx not in anomalies:
                    anomalies[idx] = []
                # Check if not already added as statistical outlier
                already_added = any(
                    issue['column'] == 'amount' and issue['type'] == 'statistical' 
                    for issue in anomalies[idx]
                )
                if not already_added:
                    anomalies[idx].append({
                        'type': 'business_rule',
                        'column': 'amount',
                        'value': float(df.loc[idx, 'amount']),
                        'issue': f'Extreme value (>{threshold_99:.2f})'
                    })
    
    # 3. Data quality issues
    if 'category' in df.columns:
        # Missing or suspicious categories
        for idx in df.index:
            cat_value = df.loc[idx, 'category']
            if pd.isna(cat_value) or cat_value in ['', None]:
                if idx not in anomalies:
                    anomalies[idx] = []
                anomalies[idx].append({
                    'type': 'data_quality',
                    'column': 'category',
                    'value': 'NULL',
                    'issue': 'Missing category'
                })
            elif cat_value in ['Unknown', 'UNKNOWN', 'Suspicious']:
                if idx not in anomalies:
                    anomalies[idx] = []
                anomalies[idx].append({
                    'type': 'data_quality',
                    'column': 'category',
                    'value': str(cat_value),
                    'issue': 'Suspicious category'
                })
    
    # 4. Pattern-based anomalies (if we have merchant data)
    if 'merchant' in df.columns:
        # Unknown merchants
        for idx in df.index:
            merchant_value = df.loc[idx, 'merchant']
            if pd.notna(merchant_value) and 'UNKNOWN' in str(merchant_value).upper():
                if idx not in anomalies:
                    anomalies[idx] = []
                anomalies[idx].append({
                    'type': 'pattern',
                    'column': 'merchant',
                    'value': str(merchant_value),
                    'issue': 'Unknown merchant'
                })
    
    return anomalies

def create_anomaly_report(df, anomalies, explainer, max_explain=10):
    """Create comprehensive anomaly report"""
    report = {
        'summary': {},
        'details': [],
        'statistics': {}
    }
    
    # Summary statistics
    report['summary'] = {
        'total_records': len(df),
        'total_anomalies': len(anomalies),
        'anomaly_rate': f"{len(anomalies)/len(df)*100:.2f}%",
        'types': {
            'statistical': 0,
            'business_rule': 0,
            'data_quality': 0,
            'pattern': 0
        }
    }
    
    # Process each anomaly
    all_anomaly_data = []
    
    for idx, issues in anomalies.items():
        record = df.iloc[idx].to_dict()
        
        # Count types
        for issue in issues:
            report['summary']['types'][issue['type']] = report['summary']['types'].get(issue['type'], 0) + 1
        
        # Create anomaly record
        anomaly_record = {
            'index': idx,
            'record': record,
            'issues': issues,
            'risk_level': 'Low'
        }
        
        # Get explanation for first N anomalies
        if len(all_anomaly_data) < max_explain:
            explanation_data = explainer.explain_anomaly_batch(
                {'record': record, 'anomaly_details': issues},
                use_llm=True
            )
            anomaly_record.update(explanation_data)
        else:
            # For non-explained anomalies, still determine risk level
            explanation_data = explainer.explain_anomaly_batch(
                {'record': record, 'anomaly_details': issues},
                use_llm=False
            )
            anomaly_record.update(explanation_data)
        
        all_anomaly_data.append(anomaly_record)
    
    # Sort by risk level
    risk_order = {'High': 0, 'Medium': 1, 'Low': 2}
    all_anomaly_data.sort(key=lambda x: risk_order.get(x.get('risk_level', 'Low'), 3))
    
    report['details'] = all_anomaly_data
    
    # Calculate risk distribution
    report['statistics']['risk_distribution'] = {
        'High': sum(1 for a in all_anomaly_data if a.get('risk_level') == 'High'),
        'Medium': sum(1 for a in all_anomaly_data if a.get('risk_level') == 'Medium'),
        'Low': sum(1 for a in all_anomaly_data if a.get('risk_level') == 'Low')
    }
    
    return report

def generate_downloadable_report(report, df):
    """Generate downloadable CSV with all anomalies - with encoding fix"""
    rows = []
    
    for anomaly in report['details']:
        row = {
            'Row_Index': anomaly['index'],
            'Risk_Level': anomaly.get('risk_level', 'Unknown'),
            'Anomaly_Types': ', '.join(anomaly.get('anomaly_types', [])).replace('σ', 'std'),
            'Explanation': str(anomaly.get('explanation', 'Not analyzed')).replace('σ', 'std'),
        }
        
        # Add record data
        for key, value in anomaly['record'].items():
            # Handle None values and convert to string
            if pd.isna(value):
                row[f'Data_{key}'] = ''
            else:
                row[f'Data_{key}'] = str(value)
        
        # Add issue details
        issues_summary = []
        for issue in anomaly['issues']:
            if 'deviation' in issue:
                issues_summary.append(f"{issue['column']}: {abs(issue['deviation']):.1f} std deviations")
            elif 'issue' in issue:
                issues_summary.append(f"{issue['column']}: {issue['issue']}")
        row['Issues'] = '; '.join(issues_summary)
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def validate_comprehensive(file, num_to_display, analyze_all):
    """Comprehensive validation with full reporting"""
    
    if file is None:
        return "Please upload a file", None, None, None, None, None
    
    try:
        # Load data
        df = pd.read_csv(file.name)
        
        # Detect all anomalies
        anomalies = detect_anomalies_comprehensive(df)
        
        # Initialize explainer
        explainer = OllamaExplainer()
        
        # Create comprehensive report
        max_explain = len(anomalies) if analyze_all else min(num_to_display, len(anomalies))
        report = create_anomaly_report(df, anomalies, explainer, max_explain)
        
        # Generate summary
        summary = f"""
# 📊 Comprehensive Data Validation Report

## Overview
- **Total Records:** {report['summary']['total_records']}
- **Anomalies Detected:** {report['summary']['total_anomalies']} ({report['summary']['anomaly_rate']})
- **Data Completeness:** {100 - (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%

## Anomaly Breakdown by Type
- **Statistical Outliers:** {report['summary']['types']['statistical']}
- **Business Rule Violations:** {report['summary']['types']['business_rule']}
- **Data Quality Issues:** {report['summary']['types']['data_quality']}
- **Pattern Anomalies:** {report['summary']['types']['pattern']}

## Risk Distribution
- **🔴 High Risk:** {report['statistics']['risk_distribution']['High']} anomalies
- **🟡 Medium Risk:** {report['statistics']['risk_distribution']['Medium']} anomalies  
- **🟢 Low Risk:** {report['statistics']['risk_distribution']['Low']} anomalies
        """
        
        # Create display dataframe (limited for UI)
        display_data = []
        for i, anomaly in enumerate(report['details'][:num_to_display]):
            display_row = {
                'Index': anomaly['index'],
                'Risk': anomaly.get('risk_level', 'Unknown'),
                'Type': ', '.join(anomaly.get('anomaly_types', [])).replace('σ', 'std'),
            }
            # Add key fields
            for key in ['transaction_id', 'amount', 'category', 'merchant']:
                if key in anomaly['record']:
                    value = anomaly['record'][key]
                    if pd.notna(value):
                        display_row[key.replace('_', ' ').title()] = value
            display_data.append(display_row)
        
        display_df = pd.DataFrame(display_data) if display_data else pd.DataFrame()
        
        # Generate detailed explanations
        explanations = "# 🤖 Detailed Anomaly Analysis\n\n"
        explanations += f"**Showing {min(num_to_display, len(report['details']))} of {len(report['details'])} anomalies**\n\n"
        
        for i, anomaly in enumerate(report['details'][:num_to_display]):
            explanations += f"## Anomaly {i+1} (Row {anomaly['index']})\n"
            
            # Risk badge
            risk = anomaly.get('risk_level', 'Unknown')
            risk_emoji = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}.get(risk, '⚪')
            explanations += f"**Risk Level:** {risk_emoji} {risk}\n\n"
            
            # Record details
            record = anomaly['record']
            for key in ['transaction_id', 'amount', 'category', 'merchant']:
                if key in record:
                    value = record[key]
                    if pd.notna(value):
                        if key == 'amount':
                            explanations += f"**{key.replace('_', ' ').title()}:** ${value:.2f}\n"
                        else:
                            explanations += f"**{key.replace('_', ' ').title()}:** {value}\n"
            
            # Issues found
            explanations += f"\n**Issues Detected:**\n"
            for issue in anomaly['issues']:
                if issue['type'] == 'statistical':
                    explanations += f"- {issue['column']}: {abs(issue['deviation']):.1f} standard deviations from mean\n"
                elif issue['type'] in ['business_rule', 'data_quality', 'pattern']:
                    explanations += f"- {issue['column']}: {issue['issue']}\n"
            
            # AI explanation
            if 'explanation' in anomaly:
                explanations += f"\n**Analysis:** {anomaly['explanation']}\n"
            
            explanations += "\n---\n\n"
        
        if len(report['details']) > num_to_display:
            explanations += f"\n📝 **Note:** {len(report['details']) - num_to_display} additional anomalies not displayed. "
            explanations += "Increase display limit or download full report to see all.\n"
        
        # Generate downloadable report
        download_df = generate_downloadable_report(report, df)
        
        # Convert to CSV for download - with proper encoding
        csv_buffer = io.StringIO()
        download_df.to_csv(csv_buffer, index=False)
        csv_str = csv_buffer.getvalue()
        
        # Statistics
        stats = f"""
## Detection Statistics
- **Total Anomalies:** {len(anomalies)}
- **Detection Rate:** {len(anomalies)/len(df)*100:.2f}%
- **Analyzed with AI:** {min(max_explain, len(anomalies))}

## Thresholds Applied
- **Statistical:** >3 standard deviations
- **High Amount:** >99th percentile
- **Categories Flagged:** NULL, Empty, Unknown, Suspicious
- **Business Rules:** Negative amounts, Unknown merchants
        """
        
        # Connection status
        status = "✅ Ollama Connected" if explainer.is_available else "⚠️ Using Rule-based Analysis"
        
        return summary, display_df, explanations, stats, csv_str, status
        
    except Exception as e:
        import traceback
        error = f"Error: {str(e)}\n{traceback.format_exc()}"
        return error, None, None, None, None, "❌ Error"

def prepare_download(csv_str):
    """Prepare CSV file for download with proper encoding - FIXED"""
    if not csv_str or not csv_str.strip():
        print("No data to download")
        return gr.File.update(visible=False)
    
    try:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"anomaly_report_{timestamp}.csv"
        
        # Clean the CSV string - replace problematic characters
        clean_csv = csv_str.replace('σ', 'std').replace('Σ', 'SUM').replace('µ', 'mu')
        
        # Write with UTF-8 BOM for Excel compatibility
        with open(filename, 'w', encoding='utf-8-sig', newline='', errors='replace') as f:
            f.write(clean_csv)
        
        print(f"✅ Report successfully saved as: {filename}")
        
        # Verify file was created
        if os.path.exists(filename):
            return gr.File.update(value=filename, visible=True)
        else:
            print(f"File not found after creation: {filename}")
            return gr.File.update(visible=False)
            
    except Exception as e:
        print(f"❌ Error saving report: {e}")
        
        # Fallback: Try saving with simpler encoding
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"anomaly_report_{timestamp}_simple.csv"
            
            # Remove all non-ASCII characters for maximum compatibility
            ascii_csv = ''.join(char if ord(char) < 128 else '?' for char in csv_str)
            
            with open(filename, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(ascii_csv)
            
            print(f"✅ Report saved with simplified encoding: {filename}")
            
            if os.path.exists(filename):
                return gr.File.update(value=filename, visible=True)
                
        except Exception as e2:
            print(f"❌ Failed completely: {e2}")
    
    return gr.File.update(visible=False)

# Gradio Interface
with gr.Blocks(title="Comprehensive Data Validation", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 Comprehensive Data Validation & Anomaly Analysis
    ### Full Dataset Analysis with AI-Powered Explanations
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            status = gr.Markdown("**Status:** Initializing...")
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
            
            with gr.Row():
                num_display = gr.Slider(
                    minimum=5, 
                    maximum=100, 
                    value=10, 
                    step=5,
                    label="Anomalies to Display",
                    info="How many to show in detail"
                )
                analyze_all = gr.Checkbox(
                    label="Analyze All with AI",
                    value=False,
                    info="AI-analyze all (slower) vs top N only"
                )
            
            validate_btn = gr.Button("🚀 Run Full Analysis", variant="primary", size="lg")
            
            download_btn = gr.Button("📥 Download Full Report", variant="secondary")
            csv_download = gr.File(label="Download Report", visible=False)
            
        with gr.Column(scale=2):
            summary = gr.Markdown()
    
    with gr.Tabs():
        with gr.Tab("📊 Anomaly Summary"):
            anomaly_table = gr.Dataframe(label="Anomalies Overview", wrap=True)
        
        with gr.Tab("🔬 Detailed Analysis"):
            detailed_analysis = gr.Markdown()
        
        with gr.Tab("📈 Statistics"):
            statistics = gr.Markdown()
    
    # Hidden CSV data holder
    csv_data = gr.Textbox(visible=False)
    
    # Main analysis
    validate_btn.click(
        fn=validate_comprehensive,
        inputs=[file_input, num_display, analyze_all],
        outputs=[summary, anomaly_table, detailed_analysis, statistics, csv_data, status]
    )
    
    # Download functionality
    download_btn.click(
        fn=prepare_download,
        inputs=[csv_data],
        outputs=[csv_download]
    )
    
    gr.Markdown("""
    ---
    ### 📋 Features:
    
    1. **Comprehensive Detection**: Statistical outliers, business rules, data quality, patterns
    2. **Risk Classification**: Automatic High/Medium/Low risk assessment
    3. **AI Analysis**: Mistral-7B explains why each anomaly matters
    4. **Full Export**: Download complete report with all anomalies
    5. **Flexible Display**: Choose how many to analyze in detail
    
    ### 🎯 Anomaly Types:
    - **Statistical**: Values >3 std deviations from mean
    - **Business Rules**: Negative amounts, extreme values
    - **Data Quality**: Missing/suspicious categories
    - **Patterns**: Unknown merchants, unusual combinations
    
    ### 🚀 Quick Start:
    ```bash
    # Terminal 1: Start Ollama
    ollama serve
    
    # Terminal 2: Pull Mistral (if needed)
    ollama pull mistral:7b-instruct
    ```
    """)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Comprehensive Data Validation System")
    print("="*60)
    print("\n✨ Features:")
    print("- Complete anomaly detection with risk classification")
    print("- AI-powered explanations using Mistral-7B")
    print("- Download full reports with all anomalies")
    print("- Fixed encoding for Windows compatibility")
    print("\n📍 Starting at: http://127.0.0.1:7860\n")
    
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)