import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from model_engine import ModelEngine
import os

# Page Config
st.set_page_config(page_title="Insider Threat Analytics", layout="wide")

# Session State Initialization
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False

def main():
    st.sidebar.title("Insider Threat Analytics")
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Setup & Data", "Dashboard"])
    
    if page == "Setup & Data":
        render_setup_page()
    elif page == "Dashboard":
        if st.session_state['model_trained']:
            render_dashboard_page()
        else:
            st.warning("Please complete the Setup first.")
            render_setup_page()

def render_setup_page():
    st.header("1. Data Setup")
    
    data_source = st.radio("Choose Data Source", ["Upload CSV", "Generate Synthetic Data"])
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload Logs CSV", type=['csv'])
        st.info("CSV must contain: `timestamp`, `user_id`, `activity_type`, `volume_mb`. Optional: `dept`, `is_insider`.")
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview:", df.head())
                if st.button("Process & Analyze"):
                    with st.spinner("Processing Data..."):
                        # Save to temp
                        os.makedirs('data', exist_ok=True)
                        df.to_csv('data/synthetic_logs.csv', index=False)
                        st.session_state['logs_df'] = df
                        run_analysis(df)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                
    elif data_source == "Generate Synthetic Data":
        if st.button("Generate & Analyze"):
            with st.spinner("Generating Data..."):
                from data_gen import generate_data
                df = generate_data()
                os.makedirs('data', exist_ok=True)
                df.to_csv('data/synthetic_logs.csv', index=False)
                st.session_state['logs_df'] = df
                st.success(f"Generated {len(df)} logs.")
                run_analysis(df)

def run_analysis(logs_df):
    st.header("2. Analysis")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. Feature Engineering
        status_text.text("Extracting Features...")
        engine = ModelEngine()
        features_df = engine.process_logs(logs_df)
        st.session_state['features_df'] = features_df
        progress_bar.progress(30)
        
        # 2. Training
        status_text.text("Training Model...")
        X_layer2, y = engine.train(features_df)
        engine.save()
        st.session_state['engine'] = engine
        st.session_state['X_layer2'] = X_layer2
        progress_bar.progress(70)
        
        # 3. Predictions
        status_text.text("Running Predictions...")
        # Prepare features for prediction
        if 'user_id' in features_df.columns:
            X_pred = features_df.set_index('user_id')
        else:
            X_pred = features_df
            
        X_pred = X_pred.drop(columns=['dept', 'is_insider'], errors='ignore')
        # Ensure columns match training (handle missing columns if any)
        # In this flow, training and prediction happen on same data, so it's fine.
        
        probs, _ = engine.predict(X_pred)
        
        results_df = features_df.copy()
        results_df['risk_score'] = probs
        results_df['risk_label'] = (probs > 0.5).astype(int)
        st.session_state['results_df'] = results_df
        
        progress_bar.progress(100)
        status_text.text("Analysis Complete!")
        st.success("Model Trained and Analysis Complete. Go to Dashboard.")
        st.session_state['model_trained'] = True
        st.session_state['data_loaded'] = True
        
    except Exception as e:
        st.error(f"Analysis Failed: {e}")
        st.exception(e)

def render_dashboard_page():
    st.header("Command Center")
    
    results_df = st.session_state['results_df']
    logs_df = st.session_state['logs_df']
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(logs_df['timestamp']):
        logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])

    engine = st.session_state['engine']
    X_layer2 = st.session_state['X_layer2']
    
    # Tabs
    tab1, tab2 = st.tabs(["Overview", "Deep Dive"])
    
    with tab1:
        # Metrics
        col1, col2, col3 = st.columns(3)
        high_risk_count = results_df[results_df['risk_score'] > 0.8].shape[0]
        avg_risk = results_df['risk_score'].mean()
        total_users = results_df.shape[0]
        
        col1.metric("High Risk Users", high_risk_count, delta_color="inverse")
        col2.metric("Average Risk Score", f"{avg_risk:.2%}")
        col3.metric("Total Users Monitored", total_users)
        
        # Treemap
        st.subheader("Risk Distribution")
        if 'dept' in results_df.columns:
            path = ['dept', 'user_id']
        else:
            path = ['user_id']
            
        fig_treemap = px.treemap(
            results_df.reset_index(), # Ensure user_id is available
            path=path,
            values='total_volume_mb',
            color='risk_score',
            color_continuous_scale='RdYlGn_r',
            title='User Risk & Volume'
        )
        st.plotly_chart(fig_treemap, use_container_width=True)
        
        # High Risk List
        st.subheader("Top High Risk Users")
        high_risk_users = results_df.sort_values('risk_score', ascending=False).head(10)
        st.dataframe(high_risk_users.style.background_gradient(subset=['risk_score'], cmap='Reds'))

    with tab2:
        st.header("Deep Dive Investigation")
        
        # User Selector
        users = results_df.index.tolist() if 'user_id' not in results_df.columns else results_df['user_id'].tolist()
        # If user_id is index
        if 'user_id' not in results_df.columns:
             results_df = results_df.reset_index()
             
        selected_user = st.selectbox("Select User", results_df['user_id'].unique())
        
        if selected_user:
            user_data = results_df[results_df['user_id'] == selected_user].iloc[0]
            user_logs = logs_df[logs_df['user_id'] == selected_user]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("User Profile")
                if 'dept' in user_data:
                    st.write(f"**Department:** {user_data['dept']}")
                st.write(f"**Risk Score:** {user_data['risk_score']:.2%}")
                
                if user_data['risk_score'] > 0.8:
                    st.error("CRITICAL RISK DETECTED")
                elif user_data['risk_score'] > 0.5:
                    st.warning("ELEVATED RISK")
                else:
                    st.success("LOW RISK")
                    
                st.subheader("Top Risk Factors")
                try:
                    user_row_layer2 = X_layer2.loc[selected_user]
                    reasons = engine.get_reason_codes(user_row_layer2)
                    for r in reasons:
                        st.write(f"- **{r['feature']}**: {r['value']:.2f} (Impact: {r['shap_value']:.2f})")
                except Exception as e:
                    st.info("Risk factors unavailable for this user.")

            with col2:
                st.subheader("Activity Timeline")
                fig_timeline = px.scatter(
                    user_logs,
                    x='timestamp',
                    y='volume_mb',
                    color='activity_type',
                    size='volume_mb',
                    title='Activity Timeline',
                    hover_data=['activity_type', 'volume_mb']
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

if __name__ == "__main__":
    main()
