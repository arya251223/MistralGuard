import pandas as pd
import io
import os
from datetime import datetime
import numpy as np

def generate_report_structure(df, anomalies, explainer, max_explain=10, progress_callback=None):
    """Create structured report data"""
    report_data = []
    
    # Sort anomalies by severity/index
    sorted_indices = sorted(anomalies.keys())
    total_anomalies = len(sorted_indices)
    
    count = 0
    for i, idx in enumerate(sorted_indices):
        issues = anomalies[idx]
        record = df.iloc[idx].to_dict()
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback((i + 1) / total_anomalies, f"Analyzing anomaly {i+1}/{total_anomalies}")

        # Get explanation
        if count < max_explain:
            analysis = explainer.explain_anomaly(record, issues)
            count += 1
        else:
            analysis = explainer._fallback_explanation(issues)
            
        entry = {
            'index': idx,
            'record': record,
            'issues': issues,
            'analysis': analysis
        }
        report_data.append(entry)
        
    return report_data

def create_csv_download(report_data):
    """Generate CSV string from report data"""
    rows = []
    for item in report_data:
        row = {
            'Row_Index': item['index'],
            'Risk_Score': item['analysis'].get('risk_score', 0),
            'Reasoning': item['analysis'].get('reasoning', ''),
            'Impact': item['analysis'].get('impact', ''),
            'Recommendation': item['analysis'].get('recommendation', '')
        }
        
        # Add record data
        for k, v in item['record'].items():
            row[f'Data_{k}'] = v
            
        # Add issues
        issues_str = "; ".join([f"{i['column']}: {i['issue']}" for i in item['issues']])
        row['Issues'] = issues_str
        
        rows.append(row)
        
    df = pd.DataFrame(rows)
    
    # CSV buffer
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()

def save_report_locally(csv_str):
    """Save report to disk"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"anomaly_report_{timestamp}.csv"
    
    # Fix encoding for Excel
    with open(filename, 'w', encoding='utf-8-sig', newline='', errors='replace') as f:
        f.write(csv_str)
        
    return filename
