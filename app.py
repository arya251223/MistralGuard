import gradio as gr
import pandas as pd
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go
from src.detector import AnomalyDetector
from src.explainer import OllamaExplainer
from src.utils import generate_report_structure, create_csv_download, save_report_locally

# Global state
detector = AnomalyDetector()
explainer = OllamaExplainer()



def save_file(csv_str):
    if not csv_str:
        return None
    filename = save_report_locally(csv_str)
    return filename

def chat_response(message, history, report_data):
    if not report_data:
        return "Please analyze a dataset first."
    return explainer.chat_with_data(message, report_data)

# UI
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"]
)

with gr.Blocks(title="MistralGuard Enhanced") as demo:
    demo.theme = theme
    gr.Markdown(
        """
        # 🛡️ MistralGuard Enhanced
        ### AI-Powered Anomaly Detection & Explanation
        """
    )
    
    # State to hold report data for chat
    report_state = gr.State()
    
    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 1. Upload Data")
            file_input = gr.File(label="Upload CSV Data", file_types=[".csv"])
            
            with gr.Accordion("⚙️ Configuration", open=False):
                threshold = gr.Slider(1.0, 5.0, value=3.0, label="Statistical Threshold (Z-Score)")
                use_ml = gr.Checkbox(value=True, label="Use Isolation Forest (ML)")
                max_explain = gr.Slider(1, 50, value=5, step=1, label="Max AI Explanations")
            
            analyze_btn = gr.Button("🚀 Analyze Data", variant="primary", size="lg")
            status_output = gr.Markdown("Ready to analyze...", visible=True)
            
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("📊 Dashboard"):
                    with gr.Row():
                        plot_risk = gr.Plot(label="Risk Distribution")
                        plot_types = gr.Plot(label="Anomaly Types")
                    plot_time = gr.Plot(label="Time Series Analysis")
                
                with gr.Tab("📋 Data View"):
                    data_output = gr.Dataframe(label="Analyzed Data", interactive=False)

                with gr.Tab("📝 Detailed Analysis"):
                    details_output = gr.Markdown()
                    
                with gr.Tab("💬 Chat with Data"):
                    chatbot = gr.ChatInterface(
                        fn=chat_response,
                        additional_inputs=[report_state],
                        title="Ask about the Anomalies",
                        description="Ask questions like 'Why is row 5 suspicious?' or 'Summarize the high risk items'."
                    )
                    
                with gr.Tab("📥 Export"):
                    gr.Markdown("### Download Report")
                    csv_store = gr.State()
                    download_btn = gr.Button("Generate Report File")
                    download_file = gr.File(label="Download CSV")
    
    def analyze_data_wrapper(file, threshold, use_ml, max_explain, progress=gr.Progress()):
        if file is None:
            return None, "Please upload a file", None, None, None, None, None, None

        try:
            progress(0, desc="Reading file...")
            df = pd.read_csv(file.name)
            
            # Detect
            progress(0.1, desc="Detecting anomalies...")
            anomalies = detector.detect_all(df, threshold=threshold, use_ml=use_ml)
            
            # Explain
            progress(0.3, desc="Generating AI explanations...")
            report_data = generate_report_structure(
                df, 
                anomalies, 
                explainer, 
                max_explain=max_explain,
                progress_callback=lambda p, msg: progress(0.3 + (p * 0.6), desc=msg)
            )
            
            progress(0.9, desc="Creating visualizations...")
            
            # Generate CSV
            csv_str = create_csv_download(report_data)
            
            # Visualizations
            # 1. Risk Distribution
            risks = [item['analysis'].get('risk_score', 0) for item in report_data]
            fig_risk = px.histogram(x=risks, nbins=10, labels={'x': 'Risk Score'}, title="Risk Score Distribution", color_discrete_sequence=['#6366f1'])
            fig_risk.update_layout(template="plotly_white")
            
            # 2. Anomaly Types
            types = []
            for item in report_data:
                for issue in item['issues']:
                    types.append(issue['type'])
            
            if types:
                fig_types = px.pie(names=types, title="Anomaly Types Breakdown", hole=0.4)
                fig_types.update_layout(template="plotly_white")
            else:
                fig_types = go.Figure()

            # 3. Time Series
            fig_time = go.Figure()
            date_col = None
            
            # Try to find a date column
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col])
                        date_col = col
                        break
                    except:
                        continue
            
            # If no date column found by name, check dtypes or just use index
            if not date_col:
                # Fallback: Use index as time axis
                x_axis = df.index
                x_label = "Record Index"
                title_suffix = "(Index-based)"
            else:
                df_sorted = df.sort_values(date_col)
                x_axis = df_sorted[date_col]
                x_label = date_col
                title_suffix = f"({date_col})"
                # Re-align df for plotting if sorted
                df = df_sorted

            # Determine Y-axis (Amount or first numeric)
            y_col = 'amount' if 'amount' in df.columns else (df.select_dtypes(include=[np.number]).columns[0] if len(df.select_dtypes(include=[np.number]).columns)>0 else None)
            
            if y_col:
                fig_time = px.line(df, x=x_axis, y=y_col, title=f"Time Series Analysis {title_suffix}")
                fig_time.update_layout(xaxis_title=x_label, template="plotly_white")
                
                # Highlight anomalies
                anomaly_indices = [item['index'] for item in report_data]
                
                # Filter anomalies that exist in the current df (in case of sorting/filtering)
                valid_indices = [i for i in anomaly_indices if i in df.index]
                
                if valid_indices:
                    anomaly_x = df.loc[valid_indices, date_col] if date_col else valid_indices
                    anomaly_y = df.loc[valid_indices, y_col]
                    
                    fig_time.add_trace(go.Scatter(
                        x=anomaly_x, 
                        y=anomaly_y, 
                        mode='markers', 
                        marker=dict(color='red', size=10, symbol='x'),
                        name='Anomalies'
                    ))
            else:
                fig_time.add_annotation(text="No numeric column found for time series", showarrow=False)

            # Summary Text
            summary = f"### ✅ Analysis Complete\n- **Total Records**: {len(df)}\n- **Anomalies Detected**: {len(anomalies)}\n- **Analyzed with AI**: {min(len(anomalies), max_explain)}"
            
            # Detailed View
            details_text = ""
            for item in report_data[:max_explain]:
                details_text += f"#### Row {item['index']} (Risk: {item['analysis'].get('risk_score')})\n"
                details_text += f"**Reasoning**: {item['analysis'].get('reasoning')}\n"
                details_text += f"**Impact**: {item['analysis'].get('impact')}\n"
                details_text += f"**Recommendation**: {item['analysis'].get('recommendation')}\n"
                details_text += "---\n"
                
            # Prepare Dataframe for display (add Anomaly flag)
            display_df = df.copy()
            display_df['Is_Anomaly'] = display_df.index.isin(anomalies.keys())
            # Move Is_Anomaly to front
            cols = ['Is_Anomaly'] + [c for c in display_df.columns if c != 'Is_Anomaly']
            display_df = display_df[cols]
                
            return csv_str, summary, fig_risk, fig_types, fig_time, details_text, report_data, display_df
        
        except Exception as e:
            import traceback
            return None, f"Error: {str(e)}\n{traceback.format_exc()}", None, None, None, None, None, None

    analyze_btn.click(
        analyze_data_wrapper,
        inputs=[file_input, threshold, use_ml, max_explain],
        outputs=[csv_store, status_output, plot_risk, plot_types, plot_time, details_output, report_state, data_output]
    )
    
    download_btn.click(
        save_file,
        inputs=[csv_store],
        outputs=[download_file]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False)
