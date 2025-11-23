"""
Weather Forecasting Dashboard - Streamlit App
Complete visualization and analysis interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from weather_analysis import WeatherAnalysisPipeline
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Weather Trend Forecasting",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced global CSS for better visibility
st.markdown("""
    <style>
    /* Global metric styling */
    div[data-testid="stMetricValue"] {
        background-color: rgba(28, 131, 225, 0.15) !important;
        padding: 20px !important;
        border-radius: 10px !important;
        border: 2px solid rgba(28, 131, 225, 0.4) !important;
        font-size: 28px !important;
        color: #0d47a1 !important;
        font-weight: bold !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #1565c0 !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.05);
        padding: 10px;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2ca02c;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== INITIALIZE SESSION STATE ====================

@st.cache_resource
def load_pipeline(data_path):
    """Load and cache the analysis pipeline"""
    pipeline = WeatherAnalysisPipeline(data_path)
    return pipeline

@st.cache_data
def run_analysis(_pipeline):
    """Run the complete analysis (cached)"""
    results, model_comparison = _pipeline.run_complete_analysis()
    return _pipeline.df_cleaned, results, model_comparison

# ==================== SIDEBAR ====================

st.sidebar.title("🌤️ Weather Forecasting")
st.sidebar.markdown("---")

# PM Accelerator Mission
st.sidebar.markdown("""
### 🎯 PM Accelerator Mission
**By making industry-leading tools and education available to individuals from all backgrounds, we level the playing field for future PM leaders.**

This is the PM Accelerator motto, as we grant aspiring and experienced PMs what they need most – **Access**. 

We introduce you to industry leaders, surround you with the right PM ecosystem, and discover the new world of AI product management skills.
""")

st.sidebar.markdown("---")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=['csv'])

if uploaded_file is not None:
    data_path = uploaded_file
else:
    data_path = 'data/Global Weather Repository.csv'
    st.sidebar.info("Using default dataset path. Upload a file to analyze custom data.")

# Navigation
st.sidebar.markdown("### 📊 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Overview", 
     "🧹 Data Processing", 
     "📈 EDA & Visualizations", 
     "🤖 Forecasting Models",
     "🎯 Advanced Analysis",
     "📊 Model Comparison",
     "📝 Summary Report"]
)

# Run analysis button
if st.sidebar.button("🚀 Run Complete Analysis", type="primary"):
    with st.spinner("Running complete analysis... This may take a few minutes..."):
        try:
            pipeline = load_pipeline(data_path)
            df_cleaned, results, model_comparison = run_analysis(pipeline)
            st.session_state['pipeline'] = pipeline
            st.session_state['df_cleaned'] = df_cleaned
            st.session_state['results'] = results
            st.session_state['model_comparison'] = model_comparison
            st.sidebar.success("✅ Analysis completed!")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")

# ==================== MAIN CONTENT ====================

# Check if analysis has been run
if 'df_cleaned' not in st.session_state:
    st.title("🌤️ Weather Trend Forecasting Dashboard")
    st.markdown("---")
    
    st.info("👈 Click **'Run Complete Analysis'** in the sidebar to start!")
    
    st.markdown("""
    ## 📋 Project Overview
    
    This dashboard provides comprehensive weather trend analysis and forecasting capabilities including:
    
    ### ✨ Features:
    - **Data Processing**: Automated cleaning, outlier detection, and feature engineering
    - **EDA**: Interactive visualizations and statistical insights
    - **Forecasting Models**: ARIMA, Prophet, LSTM, Random Forest, XGBoost
    - **Advanced Analysis**: Climate patterns, air quality, spatial analysis
    - **Model Comparison**: Performance metrics and ensemble methods
    
    ### 📊 Dataset Requirements:
    - CSV format with weather data
    - Must include `last_updated` column for time series analysis
    - Recommended columns: temperature, humidity, pressure, wind speed, etc.
    
    ### 🚀 Getting Started:
    1. Upload your dataset (or use default)
    2. Click "Run Complete Analysis"
    3. Navigate through different sections
    4. Explore insights and predictions
    """)
    
    # Display PM Accelerator Mission prominently
    st.markdown("---")
    st.markdown("""
    ### 🎯 PM Accelerator Mission
    
    **Vision**: Empowering innovation through data-driven insights and cutting-edge technology solutions.
    
    **Mission**: To accelerate project delivery and maximize impact through advanced analytics, 
    machine learning, and comprehensive data science methodologies.
    
    **Values**:
    - 📊 Data-Driven Decision Making
    - 🚀 Innovation & Excellence
    - 🤝 Collaboration & Knowledge Sharing
    - 📈 Continuous Improvement
    """)
    
    st.stop()

# Get data from session state
df_cleaned = st.session_state['df_cleaned']
results = st.session_state['results']
model_comparison = st.session_state.get('model_comparison', None)
pipeline = st.session_state['pipeline']

# ==================== PAGE: OVERVIEW ====================

if page == "🏠 Overview":
    st.title("🏠 Project Overview")
    st.markdown("---")
    
    # Key metrics with enhanced styling
    st.markdown("""
        <style>
        div[data-testid="stMetricValue"] {
            background-color: rgba(28, 131, 225, 0.15);
            padding: 20px;
            border-radius: 10px;
            border: 2px solid rgba(28, 131, 225, 0.4);
            font-size: 28px !important;
            color: #0d47a1 !important;
            font-weight: bold !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #1565c0 !important;
            font-size: 16px !important;
            font-weight: 600 !important;
        }
        </style>
        """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Total Records", f"{len(df_cleaned):,}")
    with col2:
        st.metric("🔢 Features", f"{len(df_cleaned.columns)}")
    with col3:
        outlier_pct = (df_cleaned['is_outlier'] == -1).sum() / len(df_cleaned) * 100 if 'is_outlier' in df_cleaned.columns else 0
        st.metric("⚠️ Outliers Detected", f"{outlier_pct:.2f}%")
    with col4:
        countries = df_cleaned['country'].nunique() if 'country' in df_cleaned.columns else 0
        st.metric("🌍 Countries", f"{countries}")
    
    st.markdown("---")
    
    # Dataset preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df_cleaned.head(100), use_container_width=True, height=300)
    
    # Data info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Types")
        dtype_df = pd.DataFrame({
            'Type': df_cleaned.dtypes.value_counts().index.astype(str),
            'Count': df_cleaned.dtypes.value_counts().values
        })
        fig = px.pie(dtype_df, values='Count', names='Type', title='Column Data Types')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Basic Statistics")
        st.dataframe(df_cleaned.describe().T, height=400)

# ==================== PAGE: DATA PROCESSING ====================

elif page == "🧹 Data Processing":
    st.title("🧹 Data Processing")
    st.markdown("---")
    
    st.subheader("1️⃣ Missing Values Handling")
    
    # Original vs Cleaned
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Dataset**")
        if hasattr(pipeline, 'df') and pipeline.df is not None:
            missing_orig = pipeline.df.isnull().sum()
            missing_orig = missing_orig[missing_orig > 0].sort_values(ascending=False)
            if len(missing_orig) > 0:
                fig = px.bar(x=missing_orig.values, y=missing_orig.index, orientation='h',
                           title='Missing Values (Original)', labels={'x': 'Count', 'y': 'Column'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values in original dataset!")
    
    with col2:
        st.markdown("**After Cleaning**")
        missing_cleaned = df_cleaned.isnull().sum()
        missing_cleaned = missing_cleaned[missing_cleaned > 0].sort_values(ascending=False)
        if len(missing_cleaned) > 0:
            fig = px.bar(x=missing_cleaned.values, y=missing_cleaned.index, orientation='h',
                       title='Missing Values (Cleaned)', labels={'x': 'Count', 'y': 'Column'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ All missing values handled!")
    
    st.markdown("---")
    st.subheader("2️⃣ Outlier Detection")

    if 'is_outlier' in df_cleaned.columns:
        outlier_count = (df_cleaned['is_outlier'] == -1).sum()
        inlier_count = (df_cleaned['is_outlier'] == 1).sum()
        
        # Enhanced styling for outlier metrics
        st.markdown("""
            <style>
            div[data-testid="stMetricValue"] {
                background-color: rgba(28, 131, 225, 0.15);
                padding: 20px;
                border-radius: 10px;
                border: 2px solid rgba(28, 131, 225, 0.4);
                font-size: 28px !important;
                color: #0d47a1 !important;
                font-weight: bold !important;
            }
            div[data-testid="stMetricLabel"] {
                color: #1565c0 !important;
                font-size: 16px !important;
                font-weight: 600 !important;
            }
            </style>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("⚠️ Outliers Detected", f"{outlier_count:,}", 
                    delta=f"{outlier_count/len(df_cleaned)*100:.2f}%")
        with col2:
            st.metric("✅ Inliers", f"{inlier_count:,}",
                    delta=f"{inlier_count/len(df_cleaned)*100:.2f}%")
        
        # Visualization
        outlier_df = pd.DataFrame({
            'Category': ['Inliers', 'Outliers'],
            'Count': [inlier_count, outlier_count]
        })
        fig = px.pie(outlier_df, values='Count', names='Category', 
                    title='Outlier Distribution',
                    color_discrete_map={'Inliers': '#2ca02c', 'Outliers': '#d62728'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("3️⃣ Feature Engineering")
    
    # Show new features
    new_features = ['temp_feels_diff', 'comfort_index', 'wind_x', 'wind_y', 'year', 'month', 'day', 'hour']
    existing_new = [f for f in new_features if f in df_cleaned.columns]
    
    if existing_new:
        st.success(f"✅ Created {len(existing_new)} new features!")
        st.write("**New Features:**", ", ".join(existing_new))
        
        # Show sample of new features
        st.dataframe(df_cleaned[existing_new].head(10), use_container_width=True)

# ==================== PAGE: EDA & VISUALIZATIONS ====================

elif page == "📈 EDA & Visualizations":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("---")
    
    # Temperature trends
    st.subheader("🌡️ Temperature Trends")
    if 'last_updated' in df_cleaned.columns and 'temperature_celsius' in df_cleaned.columns:
        df_sorted = df_cleaned.sort_values('last_updated').head(5000)
        fig = px.line(df_sorted, x='last_updated', y='temperature_celsius',
                     title='Temperature Over Time')
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌡️ Temperature Distribution")
        if 'temperature_celsius' in df_cleaned.columns:
            fig = px.histogram(df_cleaned, x='temperature_celsius', nbins=50,
                             title='Temperature Distribution')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💧 Humidity Distribution")
        if 'humidity' in df_cleaned.columns:
            fig = px.histogram(df_cleaned, x='humidity', nbins=50,
                             title='Humidity Distribution')
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    st.subheader("🔗 Feature Correlations")
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    key_cols = [col for col in numeric_cols if '_normalized' not in col and 'is_outlier' not in col][:15]
    
    if len(key_cols) > 0:
        corr_matrix = df_cleaned[key_cols].corr()
        fig = px.imshow(corr_matrix, 
                       title='Correlation Heatmap',
                       color_continuous_scale='RdBu_r',
                       zmin=-1, zmax=1,
                       aspect='auto')
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
    
    # Weather by country
    if 'country' in df_cleaned.columns and 'temperature_celsius' in df_cleaned.columns:
        st.subheader("🌍 Temperature by Country")
        top_countries = df_cleaned['country'].value_counts().head(10).index
        df_country = df_cleaned[df_cleaned['country'].isin(top_countries)]
        fig = px.box(df_country, x='country', y='temperature_celsius',
                    title='Temperature Distribution by Top 10 Countries')
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌡️💧 Temperature vs Humidity")
        if 'temperature_celsius' in df_cleaned.columns and 'humidity' in df_cleaned.columns:
            sample_df = df_cleaned.sample(min(5000, len(df_cleaned)))
            fig = px.scatter(sample_df, x='temperature_celsius', y='humidity',
                           title='Temperature vs Humidity', opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌡️🌬️ Temperature vs Wind Speed")
        if 'temperature_celsius' in df_cleaned.columns and 'wind_kph' in df_cleaned.columns:
            sample_df = df_cleaned.sample(min(5000, len(df_cleaned)))
            fig = px.scatter(sample_df, x='temperature_celsius', y='wind_kph',
                           title='Temperature vs Wind Speed', opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: FORECASTING MODELS ====================

elif page == "🤖 Forecasting Models":
    st.title("🤖 Forecasting Models")
    st.markdown("---")
    
    if model_comparison is not None and not isinstance(model_comparison, dict):
        st.subheader("📊 Model Performance Comparison")
    
        # Convert to DataFrame if it's a dict
        if isinstance(model_comparison, dict):
            model_comparison = pd.DataFrame(model_comparison).T
        
        # Check if DataFrame has the required columns
        if isinstance(model_comparison, pd.DataFrame) and not model_comparison.empty:
            st.dataframe(model_comparison.style.highlight_min(axis=0, subset=['RMSE', 'MAE'], color='lightgreen')
                                            .highlight_max(axis=0, subset=['R2'], color='lightgreen'),
                        use_container_width=True)
        else:
            st.dataframe(model_comparison, use_container_width=True)
        
        # Visualize comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(model_comparison.reset_index(), x='index', y='RMSE',
                        title='RMSE Comparison (Lower is Better)',
                        labels={'index': 'Model', 'RMSE': 'RMSE'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(model_comparison.reset_index(), x='index', y='R2',
                        title='R² Comparison (Higher is Better)',
                        labels={'index': 'Model', 'R2': 'R² Score'})
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Individual model details
    st.subheader("📈 Model Predictions")
    
    model_tabs = st.tabs(["ARIMA", "Prophet", "LSTM", "Random Forest", "XGBoost", "Ensemble"])
    
    # ARIMA
    with model_tabs[0]:
        if 'arima' in pipeline.predictions and pipeline.predictions['arima'] is not None:
            pred = pipeline.predictions['arima']
            st.write("**ARIMA Model Performance:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{pred['metrics']['RMSE']:.4f}")
            col2.metric("MAE", f"{pred['metrics']['MAE']:.4f}")
            col3.metric("R²", f"{pred['metrics']['R2']:.4f}")
            
            # Plot predictions
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pred['test'].index, y=pred['test'].values.flatten(),
                                    name='Actual', mode='lines'))
            fig.add_trace(go.Scatter(x=pred['test'].index, y=pred['forecast'],
                                    name='Forecast', mode='lines'))
            fig.update_layout(title='ARIMA Predictions', xaxis_title='Date', yaxis_title='Temperature')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ARIMA model not available")
    
    # Prophet
    with model_tabs[1]:
        if 'prophet' in pipeline.predictions and pipeline.predictions['prophet'] is not None:
            pred = pipeline.predictions['prophet']
            st.write("**Prophet Model Performance:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{pred['metrics']['RMSE']:.4f}")
            col2.metric("MAE", f"{pred['metrics']['MAE']:.4f}")
            col3.metric("R²", f"{pred['metrics']['R2']:.4f}")
            
            # Plot forecast
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pred['test']['ds'], y=pred['test']['y'],
                                    name='Actual', mode='lines'))
            train_size = len(pred['train'])
            forecast_test = pred['forecast'].iloc[train_size:]
            fig.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['yhat'],
                                    name='Forecast', mode='lines'))
            fig.update_layout(title='Prophet Predictions', xaxis_title='Date', yaxis_title='Temperature')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Prophet model not available")
    
    # LSTM
    with model_tabs[2]:
        if 'lstm' in pipeline.predictions and pipeline.predictions['lstm'] is not None:
            pred = pipeline.predictions['lstm']
            st.write("**LSTM Model Performance:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{pred['metrics']['RMSE']:.4f}")
            col2.metric("MAE", f"{pred['metrics']['MAE']:.4f}")
            col3.metric("R²", f"{pred['metrics']['R2']:.4f}")
            
            # Plot predictions
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=pred['y_test'].flatten(), name='Actual', mode='lines'))
            fig.add_trace(go.Scatter(y=pred['y_pred'].flatten(), name='Predicted', mode='lines'))
            fig.update_layout(title='LSTM Predictions', xaxis_title='Time Step', yaxis_title='Temperature')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("LSTM model not available")
    
    # Random Forest
    with model_tabs[3]:
        if 'ml_models' in pipeline.predictions and pipeline.predictions['ml_models'] is not None:
            pred = pipeline.predictions['ml_models']
            
            # Convert to DataFrame if needed and check for random_forest
            if isinstance(model_comparison, dict):
                temp_df = pd.DataFrame(model_comparison).T
                rf_metrics = temp_df.loc['random_forest'] if 'random_forest' in temp_df.index else None
            elif isinstance(model_comparison, pd.DataFrame):
                rf_metrics = model_comparison.loc['random_forest'] if 'random_forest' in model_comparison.index else None
            else:
                rf_metrics = None
            
            if rf_metrics is not None:
                st.write("**Random Forest Model Performance:**")
                col1, col2, col3 = st.columns(3)
                col1.metric("RMSE", f"{rf_metrics['RMSE']:.4f}")
                col2.metric("MAE", f"{rf_metrics['MAE']:.4f}")
                col3.metric("R²", f"{rf_metrics['R2']:.4f}")
                
                # Plot predictions vs actual
                fig = go.Figure()
                sample_size = min(500, len(pred['y_test']))
                fig.add_trace(go.Scatter(y=pred['y_test'][:sample_size], name='Actual', mode='lines'))
                fig.add_trace(go.Scatter(y=pred['rf_pred'][:sample_size], name='Predicted', mode='lines'))
                fig.update_layout(title='Random Forest Predictions', xaxis_title='Sample', yaxis_title='Temperature')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Random Forest model not available")
    
    # XGBoost
    with model_tabs[4]:
        if 'ml_models' in pipeline.predictions and pipeline.predictions['ml_models'] is not None:
            pred = pipeline.predictions['ml_models']
            
            # Convert to DataFrame if needed and check for xgboost
            if isinstance(model_comparison, dict):
                temp_df = pd.DataFrame(model_comparison).T
                xgb_metrics = temp_df.loc['xgboost'] if 'xgboost' in temp_df.index else None
            elif isinstance(model_comparison, pd.DataFrame):
                xgb_metrics = model_comparison.loc['xgboost'] if 'xgboost' in model_comparison.index else None
            else:
                xgb_metrics = None
            
            if xgb_metrics is not None:
                st.write("**XGBoost Model Performance:**")
                col1, col2, col3 = st.columns(3)
                col1.metric("RMSE", f"{xgb_metrics['RMSE']:.4f}")
                col2.metric("MAE", f"{xgb_metrics['MAE']:.4f}")
                col3.metric("R²", f"{xgb_metrics['R2']:.4f}")
                
                # Plot predictions vs actual
                fig = go.Figure()
                sample_size = min(500, len(pred['y_test']))
                fig.add_trace(go.Scatter(y=pred['y_test'][:sample_size], name='Actual', mode='lines'))
                fig.add_trace(go.Scatter(y=pred['xgb_pred'][:sample_size], name='Predicted', mode='lines'))
                fig.update_layout(title='XGBoost Predictions', xaxis_title='Sample', yaxis_title='Temperature')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("XGBoost model not available")
    
    # Ensemble
    with model_tabs[5]:
        if 'ensemble' in pipeline.predictions and pipeline.predictions['ensemble'] is not None:
            pred = pipeline.predictions['ensemble']
            st.write("**Ensemble Model Performance:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{pred['metrics']['RMSE']:.4f}")
            col2.metric("MAE", f"{pred['metrics']['MAE']:.4f}")
            col3.metric("R²", f"{pred['metrics']['R2']:.4f}")
            
            # Plot predictions vs actual
            fig = go.Figure()
            sample_size = min(500, len(pred['y_test']))
            fig.add_trace(go.Scatter(y=pred['y_test'][:sample_size], name='Actual', mode='lines'))
            fig.add_trace(go.Scatter(y=pred['predictions'][:sample_size], name='Ensemble Prediction', mode='lines'))
            fig.update_layout(title='Ensemble Predictions', xaxis_title='Sample', yaxis_title='Temperature')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Ensemble model not available")

# ==================== PAGE: ADVANCED ANALYSIS ====================

elif page == "🎯 Advanced Analysis":
    st.title("🎯 Advanced Analysis")
    st.markdown("---")
    
    # Climate Analysis
    st.subheader("🌍 Climate Pattern Analysis")
    if 'climate_analysis' in results:
        climate = results['climate_analysis']
        
        # Monthly patterns
        if 'monthly_patterns' in climate:
            st.write("**Monthly Temperature Patterns**")
            fig = go.Figure()
            monthly = climate['monthly_patterns']
            fig.add_trace(go.Scatter(x=monthly.index, y=monthly['mean'], name='Mean',
                                    mode='lines+markers'))
            fig.add_trace(go.Scatter(x=monthly.index, y=monthly['max'], name='Max',
                                    mode='lines', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=monthly.index, y=monthly['min'], name='Min',
                                    mode='lines', line=dict(dash='dash')))
            fig.update_layout(title='Monthly Temperature Patterns', 
                            xaxis_title='Month', yaxis_title='Temperature (°C)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal patterns
        if 'seasonal_patterns' in climate:
            st.write("**Seasonal Temperature Statistics**")
            st.dataframe(climate['seasonal_patterns'], use_container_width=True)
        
        # Country climate
        if 'country_climate' in climate:
            st.write("**Climate by Country (Top 15)**")
            country_climate = climate['country_climate'].reset_index()
            fig = px.bar(country_climate, x='country', y='mean',
                        error_y='std',
                        title='Average Temperature by Country',
                        labels={'mean': 'Avg Temperature (°C)', 'country': 'Country'})
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Air Quality Analysis
    st.subheader("💨 Air Quality Correlation Analysis")
    if 'air_quality' in results and results['air_quality'] is not None:
        aq_analysis = results['air_quality']
        
        # Create correlation dataframe
        aq_df = pd.DataFrame(aq_analysis).T
        st.dataframe(aq_df, use_container_width=True)
        
        # Visualize correlations
        fig = px.imshow(aq_df, 
                       title='Air Quality vs Weather Parameters Correlation',
                       color_continuous_scale='RdBu_r',
                       zmin=-1, zmax=1,
                       aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No air quality data available in dataset")
    
    st.markdown("---")
    
    # Feature Importance
    st.subheader("🎯 Feature Importance Analysis")
    if 'feature_importance' in results:
        feat_imp = results['feature_importance']
        fig = px.bar(feat_imp, x='importance', y='feature', orientation='h',
                    title='Feature Importance (Random Forest)',
                    labels={'importance': 'Importance Score', 'feature': 'Feature'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Spatial Analysis
    st.subheader("🗺️ Geographical Pattern Analysis")
    if 'spatial_analysis' in results:
        spatial = results['spatial_analysis']
        
        if 'country_stats' in spatial:
            st.write("**Weather Statistics by Country**")
            country_stats = spatial['country_stats'].head(20)
            st.dataframe(country_stats, use_container_width=True)
        
        if 'latitude_temperature' in spatial:
            st.write("**Temperature by Latitude Bands**")
            lat_temp = spatial['latitude_temperature'].reset_index()
            lat_temp['lat_band'] = lat_temp['lat_band'].astype(str)
            fig = px.bar(lat_temp, x='lat_band', y='temperature_celsius',
                        title='Temperature by Latitude Bands')
            st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: MODEL COMPARISON ====================

elif page == "📊 Model Comparison":
    st.title("📊 Comprehensive Model Comparison")
    st.markdown("---")
    
    if model_comparison is not None and len(model_comparison) > 0:
        st.subheader("🏆 Performance Metrics")
        
        # Convert to DataFrame if needed
        if isinstance(model_comparison, dict):
            model_comparison = pd.DataFrame(model_comparison).T
        
        # Check if we have RMSE column
        if isinstance(model_comparison, pd.DataFrame) and 'RMSE' in model_comparison.columns:
            best_model = model_comparison['RMSE'].idxmin()
            st.success(f"🏆 Best Model: **{best_model.upper()}** (Lowest RMSE)")
        else:
            st.warning("Model comparison data is incomplete")
            st.dataframe(model_comparison, use_container_width=True)
            st.stop()
        
        # Display metrics
        st.dataframe(model_comparison.style.highlight_min(axis=0, subset=['RMSE', 'MAE'], color='lightgreen')
                                         .highlight_max(axis=0, subset=['R2'], color='lightgreen'),
                    use_container_width=True)
        
        st.markdown("---")
        
        # Visualizations
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.bar(model_comparison.reset_index(), x='index', y='RMSE',
                        title='RMSE Comparison',
                        labels={'index': 'Model', 'RMSE': 'RMSE'},
                        color='RMSE',
                        color_continuous_scale='Reds_r')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(model_comparison.reset_index(), x='index', y='MAE',
                        title='MAE Comparison',
                        labels={'index': 'Model', 'MAE': 'MAE'},
                        color='MAE',
                        color_continuous_scale='Oranges_r')
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = px.bar(model_comparison.reset_index(), x='index', y='R2',
                        title='R² Score Comparison',
                        labels={'index': 'Model', 'R2': 'R² Score'},
                        color='R2',
                        color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Radar chart comparison
        st.subheader("📊 Multi-dimensional Model Comparison")
        
        # Normalize metrics for radar chart
        comparison_norm = model_comparison.copy()
        comparison_norm['RMSE_norm'] = 1 - (comparison_norm['RMSE'] - comparison_norm['RMSE'].min()) / (comparison_norm['RMSE'].max() - comparison_norm['RMSE'].min())
        comparison_norm['MAE_norm'] = 1 - (comparison_norm['MAE'] - comparison_norm['MAE'].min()) / (comparison_norm['MAE'].max() - comparison_norm['MAE'].min())
        comparison_norm['R2_norm'] = comparison_norm['R2']
        
        fig = go.Figure()
        
        for model in comparison_norm.index:
            fig.add_trace(go.Scatterpolar(
                r=[comparison_norm.loc[model, 'RMSE_norm'],
                   comparison_norm.loc[model, 'MAE_norm'],
                   comparison_norm.loc[model, 'R2_norm']],
                theta=['RMSE (inv)', 'MAE (inv)', 'R²'],
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title='Model Performance Radar Chart (Higher is Better)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Model insights
        st.subheader("🔍 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Strengths:**")
            for model in model_comparison.index[:3]:
                metrics = model_comparison.loc[model]
                st.write(f"- **{model.upper()}**: R² = {metrics['R2']:.4f}")
        
        with col2:
            st.markdown("**Recommendations:**")
            if best_model == 'ensemble':
                st.write("✅ Ensemble model provides best overall performance")
            elif 'lstm' in model_comparison.index:
                st.write("✅ LSTM captures temporal dependencies well")
            elif 'xgboost' in model_comparison.index:
                st.write("✅ XGBoost handles non-linear patterns effectively")
            
            st.write("✅ Consider ensemble for production deployment")
            st.write("✅ Multiple models provide robust predictions")
    else:
        st.warning("No model comparison data available. Run the analysis first!")

# ==================== PAGE: SUMMARY REPORT ====================

elif page == "📝 Summary Report":
    st.title("📝 Comprehensive Summary Report")
    st.markdown("---")
    
    # PM Accelerator Mission
    st.markdown("""
    ### 🎯 PM Accelerator Mission

    **By making industry-leading tools and education available to individuals from all backgrounds, we level the playing field for future PM leaders.**

    This is the PM Accelerator motto, as we grant aspiring and experienced PMs what they need most – **Access**. 

    We introduce you to industry leaders, surround you with the right PM ecosystem, and discover the new world of AI product management skills.

    ---
    """)
    
    # Executive Summary with better visibility
    st.subheader("📊 Executive Summary")

    # Add enhanced custom styling
    st.markdown("""
        <style>
        div[data-testid="stMetricValue"] {
            background-color: rgba(28, 131, 225, 0.15) !important;
            padding: 20px !important;
            border-radius: 10px !important;
            border: 2px solid rgba(28, 131, 225, 0.4) !important;
            font-size: 32px !important;
            color: #0d47a1 !important;
            font-weight: bold !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #1565c0 !important;
            font-size: 18px !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricDelta"] {
            color: #0d47a1 !important;
        }
        </style>
        """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Total Records", f"{len(df_cleaned):,}")
    with col2:
        st.metric("🔢 Features Analyzed", f"{len(df_cleaned.columns)}")
    with col3:
        # Convert to DataFrame if dict to get length
        if isinstance(model_comparison, dict):
            models_count = len(model_comparison)
        elif isinstance(model_comparison, pd.DataFrame):
            models_count = len(model_comparison)
        else:
            models_count = 0
        st.metric("🤖 Models Built", f"{models_count}")
    with col4:
        countries = df_cleaned['country'].nunique() if 'country' in df_cleaned.columns else 0
        st.metric("🌍 Countries Covered", f"{countries}")
    
    st.markdown("---")
    
    # Data Quality Report
    st.subheader("🧹 Data Quality Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Data Completeness:**")
        total_cells = df_cleaned.shape[0] * df_cleaned.shape[1]
        missing_cells = df_cleaned.isnull().sum().sum()
        completeness = (1 - missing_cells/total_cells) * 100
        
        st.markdown(f"""
        <div style='background-color: rgba(28, 131, 225, 0.1); 
                    padding: 20px; 
                    border-radius: 10px; 
                    border: 1px solid rgba(28, 131, 225, 0.3);'>
            <h2 style='color: #1c83e1; margin: 0;'>{completeness:.2f}%</h2>
            <p style='color: #0e4c92; margin: 5px 0 0 0;'>Data Completeness</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Outlier Treatment:**")
        if 'is_outlier' in df_cleaned.columns:
            outlier_pct = (df_cleaned['is_outlier'] == -1).sum() / len(df_cleaned) * 100
            st.write(f"- Outliers detected: {outlier_pct:.2f}%")
            st.write(f"- Treatment: Statistical capping (3σ)")
        
    with col2:
        st.markdown("**Feature Engineering:**")
        new_features = ['temp_feels_diff', 'comfort_index', 'wind_x', 'wind_y', 
                       'year', 'month', 'day', 'hour', 'is_weekend']
        created = [f for f in new_features if f in df_cleaned.columns]
        st.write(f"- New features created: {len(created)}")
        st.write(f"- Total features: {len(df_cleaned.columns)}")
        
        st.markdown("**Data Processing:**")
        st.write("✅ Missing values handled")
        st.write("✅ Outliers detected & treated")
        st.write("✅ Features engineered")
        st.write("✅ Data normalized")
    
    st.markdown("---")
    
    # EDA Insights
    st.subheader("📈 Key EDA Findings")

    # Keep the same enhanced styling from above
    if 'temperature_celsius' in df_cleaned.columns:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🌡️ Avg Temperature", f"{df_cleaned['temperature_celsius'].mean():.2f}°C")
        with col2:
            st.metric("🔥 Max Temperature", f"{df_cleaned['temperature_celsius'].max():.2f}°C")
        with col3:
            st.metric("❄️ Min Temperature", f"{df_cleaned['temperature_celsius'].min():.2f}°C")
    
    # Correlations
    if 'eda_insights' in results and 'temp_correlations' in results['eda_insights']:
        st.markdown("**Top Temperature Correlations:**")
        temp_corr = results['eda_insights']['temp_correlations'].head(6)
        corr_df = pd.DataFrame({
            'Feature': temp_corr.index,
            'Correlation': temp_corr.values
        })
        fig = px.bar(corr_df[1:], x='Feature', y='Correlation',
                    title='Features Most Correlated with Temperature')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Model Performance Summary
    st.subheader("🤖 Model Performance Summary")

    # Convert to DataFrame if needed
    if isinstance(model_comparison, dict):
        model_comparison_df = pd.DataFrame(model_comparison).T
    elif isinstance(model_comparison, pd.DataFrame):
        model_comparison_df = model_comparison
    else:
        model_comparison_df = None

    if model_comparison_df is not None and not model_comparison_df.empty and 'RMSE' in model_comparison_df.columns:
        best_model = model_comparison_df['RMSE'].idxmin()
        best_metrics = model_comparison_df.loc[best_model]
        
        st.success(f"🏆 **Best Performing Model: {best_model.upper()}**")
        
        # Enhanced styling still active from above
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📉 RMSE", f"{best_metrics['RMSE']:.4f}")
        with col2:
            st.metric("📊 MAE", f"{best_metrics['MAE']:.4f}")
        with col3:
            st.metric("🎯 R² Score", f"{best_metrics['R2']:.4f}")
        
        st.markdown("**All Models Performance:**")
        st.dataframe(model_comparison_df, use_container_width=True)
    else:
        st.warning("⚠️ Model comparison data is not available. Please run the complete analysis.")
        st.stop()
    
    st.markdown("---")
    
    # Advanced Analysis Summary
    st.subheader("🎯 Advanced Analysis Insights")
    
    tab1, tab2, tab3 = st.tabs(["Climate Patterns", "Feature Importance", "Spatial Analysis"])
    
    with tab1:
        if 'climate_analysis' in results:
            st.markdown("**Climate Pattern Findings:**")
            
            if 'seasonal_patterns' in results['climate_analysis']:
                seasonal = results['climate_analysis']['seasonal_patterns']
                st.write("**Temperature by Season:**")
                st.dataframe(seasonal[['mean', 'std', 'min', 'max']], use_container_width=True)
            
            st.markdown("""
            **Key Insights:**
            - Seasonal variations detected across regions
            - Temperature patterns show clear monthly trends
            - Regional climate differences identified
            """)
    
    with tab2:
        if 'feature_importance' in results:
            st.markdown("**Most Important Features for Prediction:**")
            feat_imp = results['feature_importance'].head(10)
            fig = px.bar(feat_imp, x='importance', y='feature', orientation='h',
                        title='Top 10 Most Important Features')
            st.plotly_chart(fig, use_container_width=True)
            
            top_feature = feat_imp.iloc[0]['feature']
            st.info(f"🎯 **{top_feature}** is the most important feature for temperature prediction")
    
    with tab3:
        if 'spatial_analysis' in results:
            st.markdown("**Geographical Pattern Summary:**")
            
            if 'country_stats' in results['spatial_analysis']:
                country_stats = results['spatial_analysis']['country_stats']
                st.write(f"- Analyzed data from {len(country_stats)} countries")
                st.write("- Significant geographical variations observed")
                st.write("- Climate zones clearly differentiated")
            
            st.markdown("""
            **Spatial Insights:**
            - Weather patterns vary significantly by geography
            - Latitude strongly influences temperature
            - Regional clusters identified
            """)
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("💡 Recommendations & Next Steps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Deployment:**")
        st.write("1. Deploy ensemble model for production")
        st.write("2. Implement real-time data pipeline")
        st.write("3. Set up monitoring and alerts")
        st.write("4. Regular model retraining schedule")
        
        st.markdown("**Data Collection:**")
        st.write("1. Expand temporal coverage")
        st.write("2. Add more geographical regions")
        st.write("3. Include additional weather parameters")
        st.write("4. Integrate real-time data sources")
    
    with col2:
        st.markdown("**Analysis Enhancement:**")
        st.write("1. Deep learning architectures (Transformers)")
        st.write("2. Multi-variate forecasting")
        st.write("3. Extreme weather event prediction")
        st.write("4. Climate change trend analysis")
        
        st.markdown("**Business Applications:**")
        st.write("1. Agricultural planning support")
        st.write("2. Energy demand forecasting")
        st.write("3. Transportation optimization")
        st.write("4. Emergency response planning")
    
    st.markdown("---")
    
    # Conclusion
    st.subheader("📌 Conclusion")
    
    st.markdown(f"""
    This comprehensive weather forecasting analysis has successfully:
    
    ✅ **Processed and cleaned {len(df_cleaned):,} weather records** with robust data quality measures
    
    ✅ **Built and compared {len(model_comparison) if model_comparison is not None else 0} forecasting models** using state-of-the-art techniques
    
    ✅ **Achieved {best_metrics['R2']:.2%} R² score** with the best performing {best_model.upper()} model
    
    ✅ **Conducted advanced analyses** including climate patterns, feature importance, and spatial analysis
    
    ✅ **Generated actionable insights** for weather prediction and decision-making
    
    The analysis demonstrates the power of combining multiple modeling approaches with domain-specific 
    feature engineering and comprehensive exploratory analysis. The ensemble approach provides robust 
    predictions suitable for real-world deployment.
    
    ---
    
    ### 🎯 PM Accelerator Mission Statement
    
    *"Empowering innovation through data-driven insights and cutting-edge technology solutions. 
    Building the future, one prediction at a time."*
    
    ---
    
    **Project Team**: Data Science & Analytics Division  
    **Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}  
    **Status**: ✅ Complete
    """)
    
    # Download section
    st.markdown("---")
    st.subheader("📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download Cleaned Data"):
            csv = df_cleaned.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="cleaned_weather_data.csv",
                mime="text/csv"
            )
    
    with col2:
        if model_comparison is not None:
            if st.button("📈 Download Model Metrics"):
                csv = model_comparison.to_csv()
                st.download_button(
                    label="Download Metrics",
                    data=csv,
                    file_name="model_comparison.csv",
                    mime="text/csv"
                )
    
    with col3:
        if 'feature_importance' in results:
            if st.button("🎯 Download Feature Importance"):
                csv = results['feature_importance'].to_csv(index=False)
                st.download_button(
                    label="Download Features",
                    data=csv,
                    file_name="feature_importance.csv",
                    mime="text/csv"
                )

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🌤️ Weather Trend Forecasting Dashboard | Built with Streamlit</p>
    <p>🎯 PM Accelerator - Empowering Innovation Through Data Science</p>
</div>
""", unsafe_allow_html=True)