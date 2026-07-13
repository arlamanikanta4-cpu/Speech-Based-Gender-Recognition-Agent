import streamlit as st
import pandas as pd
import os
from voice_agent import VoiceAnalysisAgent
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Speech-Based Gender Recognition Agent",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.4);
    }
    .report-card {
        background-color: #1f2937;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }
    .gender-male {
        color: #3b82f6;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .gender-female {
        color: #ec4899;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 600;
        color: #10b981;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.title("🎙️ Speech-Based Gender Recognition Agent")
st.markdown("---")

# Initialize Agent
@st.cache_resource
def get_agent():
    return VoiceAnalysisAgent()

agent = get_agent()

# Sidebar Info
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/microphone.png", width=120)
    st.markdown("### About the Agent")
    st.write(
        "This agent uses a Random Forest Classifier trained on acoustic properties of voices "
        "to determine gender. The features include frequencies, entropy, spectral properties, and frequency ranges."
    )
    st.markdown("---")
    st.markdown("### System Log Info")
    log_file_path = "logs/interaction_log.csv"
    if os.path.exists(log_file_path):
        df_logs = pd.read_csv(log_file_path)
        st.metric("Total Analyses Run", len(df_logs))
    else:
        st.metric("Total Analyses Run", 0)

# Layout: Main columns
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.subheader("📁 Upload Voice Data (CSV)")
    uploaded_file = st.file_uploader(
        "Upload a voice acoustic features CSV file",
        type=["csv"],
        help="Upload a CSV file containing acoustic voice features (e.g., meanfreq, sd, median, etc.)"
    )

    if uploaded_file is not None:
        # Display preview of uploaded data
        df_preview = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        
        with st.expander("🔍 Preview Acoustic Features Data", expanded=True):
            st.dataframe(df_preview.head(5), use_container_width=True)
            st.caption(f"Showing first 5 rows of {len(df_preview)} rows × {len(df_preview.columns)} columns")

        # Save to temp file for the agent to process
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Analysis Trigger
        st.markdown("<br>", unsafe_allow_html=True)
        run_analysis = st.button("🚀 Analyze Voice and Predict Gender", type="primary")

        if run_analysis:
            with st.spinner("Analyzing acoustic features..."):
                try:
                    # Run predictor through agent
                    result = agent.analyze(temp_file_path)
                    
                    st.session_state['analysis_result'] = result
                    st.session_state['uploaded_file_name'] = uploaded_file.name
                    
                    # Clean up temp file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                except Exception as e:
                    st.error(f"Error analyzing file: {str(e)}")

with col2:
    st.subheader("🎯 Analysis Results")
    
    if 'analysis_result' in st.session_state:
        res = st.session_state['analysis_result']
        filename = st.session_state['uploaded_file_name']
        
        gender = res.get("Gender", "Unknown")
        confidence = res.get("Confidence", "0%")
        report_saved_path = res.get("Report Saved", "")

        # Card style output
        st.markdown(f"""
            <div class="report-card">
                <h3>Prediction Summary</h3>
                <p><strong>File Name:</strong> {filename}</p>
                <p>Predicted Gender:</p>
                <div class="{'gender-male' if gender.lower() == 'male' else 'gender-female'}">
                    {gender.upper()} {'♂️' if gender.lower() == 'male' else '♀️'}
                </div>
                <br>
                <p>Confidence Level:</p>
                <div class="metric-val">{confidence}</div>
            </div>
        """, unsafe_allow_html=True)

        # Progress bar for confidence
        try:
            conf_val = float(confidence.replace("%", "")) / 100.0
            st.progress(conf_val, text=f"Confidence Metric: {confidence}")
        except ValueError:
            pass

        # Load and provide download of the written text report
        if os.path.exists(report_saved_path):
            with open(report_saved_path, "r") as report_file:
                report_content = report_file.read()
            
            st.download_button(
                label="📥 Download Detailed Report (.txt)",
                data=report_content,
                file_name=os.path.basename(report_saved_path),
                mime="text/plain"
            )
    else:
        st.info("Upload a CSV file and click **Analyze Voice and Predict Gender** to see the results.")

# Log History Section
st.markdown("---")
st.subheader("📜 Interaction & Analysis History Log")

if os.path.exists(log_file_path):
    try:
        df_logs = pd.read_csv(log_file_path)
        # Sort logs by Date descending if available
        if "Date" in df_logs.columns:
            df_logs = df_logs.sort_values(by="Date", ascending=False)
        st.dataframe(df_logs, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load history logs: {str(e)}")
else:
    st.write("No logs recorded yet. Run your first analysis to start logging.")
