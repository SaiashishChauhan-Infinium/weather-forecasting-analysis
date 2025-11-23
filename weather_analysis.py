"""
Complete Weather Analysis Pipeline
Handles: Data Processing, EDA, Forecasting, Advanced Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Time series imports
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
# try:
# except:
#     from fbprophet import Prophet

# Deep learning
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Other imports
import xgboost as xgb
from scipy import stats
import joblib
from datetime import datetime, timedelta


class WeatherAnalysisPipeline:
    """Complete weather analysis pipeline"""
    
    def __init__(self, data_path):
        self.data_path = r"C:\Users\Saiashish\Desktop\weather_forecasting\data\GlobalWeatherRepository.csv"#data_path
        self.df = None
        self.df_cleaned = None
        self.models = {}
        self.predictions = {}
        self.results = {}
        
    # ==================== DATA LOADING & PREPROCESSING ====================
    
    def load_data(self):
        """Load and initial exploration"""
        print("📂 Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        # Convert last_updated to datetime
        if 'last_updated' in self.df.columns:
            self.df['last_updated'] = pd.to_datetime(self.df['last_updated'], errors='coerce')
        
        return self.df
    
    def clean_data(self):
        """Complete data cleaning pipeline"""
        print("\n🧹 Cleaning data...")
        
        if self.df is None:
            self.load_data()
        
        self.df_cleaned = self.df.copy()
        
        # Handle missing values
        print("  → Handling missing values...")
        numeric_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df_cleaned.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            if self.df_cleaned[col].isnull().sum() > 0:
                self.df_cleaned[col].fillna(self.df_cleaned[col].median(), inplace=True)
        
        for col in categorical_cols:
            if self.df_cleaned[col].isnull().sum() > 0:
                mode_val = self.df_cleaned[col].mode()[0] if not self.df_cleaned[col].mode().empty else 'Unknown'
                self.df_cleaned[col].fillna(mode_val, inplace=True)
        
        # Detect and handle outliers
        print("  → Detecting outliers...")
        numeric_cols = [col for col in numeric_cols if 'id' not in col.lower()]
        
        if len(numeric_cols) > 0:
            X = self.df_cleaned[numeric_cols].fillna(0)
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            outlier_labels = iso_forest.fit_predict(X)
            self.df_cleaned['is_outlier'] = outlier_labels
            
            outlier_count = (outlier_labels == -1).sum()
            print(f"  → Found {outlier_count} outliers ({outlier_count/len(self.df_cleaned)*100:.2f}%)")
            
            # Cap outliers instead of removing
            for col in numeric_cols:
                mean = self.df_cleaned[col].mean()
                std = self.df_cleaned[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                self.df_cleaned[col] = np.clip(self.df_cleaned[col], lower_bound, upper_bound)
        
        # Feature engineering
        print("  → Engineering features...")
        self._create_features()
        
        print("✅ Data cleaning completed!")
        return self.df_cleaned
    
    def _create_features(self):
        """Create additional features"""
        # DateTime features
        if 'last_updated' in self.df_cleaned.columns:
            self.df_cleaned['year'] = self.df_cleaned['last_updated'].dt.year
            self.df_cleaned['month'] = self.df_cleaned['last_updated'].dt.month
            self.df_cleaned['day'] = self.df_cleaned['last_updated'].dt.day
            self.df_cleaned['day_of_week'] = self.df_cleaned['last_updated'].dt.dayofweek
            self.df_cleaned['hour'] = self.df_cleaned['last_updated'].dt.hour
            self.df_cleaned['is_weekend'] = self.df_cleaned['day_of_week'].isin([5, 6]).astype(int)
        
        # Weather-based features
        if 'temperature_celsius' in self.df_cleaned.columns and 'feels_like_celsius' in self.df_cleaned.columns:
            self.df_cleaned['temp_feels_diff'] = self.df_cleaned['temperature_celsius'] - self.df_cleaned['feels_like_celsius']
        
        if 'temperature_celsius' in self.df_cleaned.columns and 'humidity' in self.df_cleaned.columns:
            self.df_cleaned['comfort_index'] = (self.df_cleaned['temperature_celsius'] * 0.7) - (self.df_cleaned['humidity'] * 0.3)
        
        if 'wind_kph' in self.df_cleaned.columns and 'wind_degree' in self.df_cleaned.columns:
            self.df_cleaned['wind_kph'].fillna(0, inplace=True)
            self.df_cleaned['wind_degree'].fillna(0, inplace=True)
            self.df_cleaned['wind_x'] = self.df_cleaned['wind_kph'] * np.cos(np.radians(self.df_cleaned['wind_degree']))
            self.df_cleaned['wind_y'] = self.df_cleaned['wind_kph'] * np.sin(np.radians(self.df_cleaned['wind_degree']))
    
    # ==================== EXPLORATORY DATA ANALYSIS ====================
    
    def generate_eda_insights(self):
        """Generate comprehensive EDA insights"""
        print("\n📊 Generating EDA insights...")
        
        if self.df_cleaned is None:
            self.clean_data()
        
        insights = {
            'summary_stats': self.df_cleaned.describe(),
            'missing_values': self.df_cleaned.isnull().sum(),
            'data_types': self.df_cleaned.dtypes,
            'shape': self.df_cleaned.shape,
            'numeric_columns': self.df_cleaned.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.df_cleaned.select_dtypes(include=['object']).columns.tolist()
        }
        
        # Correlation analysis
        numeric_df = self.df_cleaned.select_dtypes(include=[np.number])
        insights['correlation_matrix'] = numeric_df.corr()
        
        # Top correlations with temperature
        if 'temperature_celsius' in numeric_df.columns:
            temp_corr = numeric_df.corr()['temperature_celsius'].sort_values(ascending=False)
            insights['temp_correlations'] = temp_corr
        
        self.results['eda_insights'] = insights
        print("✅ EDA insights generated!")
        return insights
    
    def create_visualizations(self):
        """Create all visualizations"""
        print("\n📈 Creating visualizations...")
        
        if self.df_cleaned is None:
            self.clean_data()
        
        viz = {}
        
        # 1. Temperature trends
        if 'last_updated' in self.df_cleaned.columns and 'temperature_celsius' in self.df_cleaned.columns:
            df_sorted = self.df_cleaned.sort_values('last_updated').head(1000)
            viz['temp_trend'] = px.line(df_sorted, x='last_updated', y='temperature_celsius',
                                       title='Temperature Trends Over Time')
        
        # 2. Correlation heatmap
        numeric_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns
        key_cols = [col for col in numeric_cols if '_normalized' not in col and 'is_outlier' not in col][:15]
        corr_matrix = self.df_cleaned[key_cols].corr()
        viz['correlation'] = px.imshow(corr_matrix, title='Feature Correlation Matrix',
                                       color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        
        # 3. Temperature distribution
        if 'temperature_celsius' in self.df_cleaned.columns:
            viz['temp_dist'] = px.histogram(self.df_cleaned, x='temperature_celsius',
                                           title='Temperature Distribution', nbins=50)
        
        # 4. Weather by country
        if 'country' in self.df_cleaned.columns and 'temperature_celsius' in self.df_cleaned.columns:
            top_countries = self.df_cleaned['country'].value_counts().head(10).index
            df_country = self.df_cleaned[self.df_cleaned['country'].isin(top_countries)]
            viz['country_temp'] = px.box(df_country, x='country', y='temperature_celsius',
                                        title='Temperature Distribution by Country')
        
        # 5. Humidity vs Temperature
        if 'humidity' in self.df_cleaned.columns and 'temperature_celsius' in self.df_cleaned.columns:
            sample_df = self.df_cleaned.sample(min(5000, len(self.df_cleaned)))
            viz['humidity_temp'] = px.scatter(sample_df, x='temperature_celsius', y='humidity',
                                             title='Humidity vs Temperature', opacity=0.5)
        
        self.results['visualizations'] = viz
        print("✅ Visualizations created!")
        return viz
    
    # ==================== FORECASTING MODELS ====================
    
    def prepare_timeseries_data(self, target_col='temperature_celsius', lookback=24):
        """Prepare data for time series forecasting"""
        if self.df_cleaned is None:
            self.clean_data()
        
        # Sort by date
        df_ts = self.df_cleaned.sort_values('last_updated').copy()
        
        # Remove duplicates and set index
        df_ts = df_ts.drop_duplicates(subset=['last_updated'])
        df_ts.set_index('last_updated', inplace=True)
        
        # Resample to hourly if needed
        if target_col in df_ts.columns:
            df_ts = df_ts[[target_col]].resample('H').mean()
            df_ts = df_ts.fillna(method='ffill').fillna(method='bfill')
        
        return df_ts
    
    def build_arima_model(self, target_col='temperature_celsius'):
        """Build ARIMA forecasting model"""
        print("\n🤖 Building ARIMA model...")
        
        try:
            df_ts = self.prepare_timeseries_data(target_col)
            
            # Split data
            train_size = int(len(df_ts) * 0.8)
            train, test = df_ts[:train_size], df_ts[train_size:]
            
            # Build ARIMA model
            model = ARIMA(train[target_col], order=(5, 1, 2))
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(steps=len(test))
            
            # Evaluate
            mse = mean_squared_error(test[target_col], forecast)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test[target_col], forecast)
            r2 = r2_score(test[target_col], forecast)
            
            self.models['arima'] = fitted_model
            self.predictions['arima'] = {
                'train': train,
                'test': test,
                'forecast': forecast,
                'metrics': {'RMSE': rmse, 'MAE': mae, 'R2': r2}
            }
            
            print(f"  ✅ ARIMA - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            return fitted_model, {'RMSE': rmse, 'MAE': mae, 'R2': r2}
            
        except Exception as e:
            print(f"  ⚠️ ARIMA failed: {str(e)}")
            return None, None
    
    def build_prophet_model(self, target_col='temperature_celsius'):
        """Build Prophet forecasting model"""
        print("\n🤖 Building Prophet model...")
        
        try:
            df_ts = self.prepare_timeseries_data(target_col)
            df_prophet = df_ts.reset_index()
            df_prophet.columns = ['ds', 'y']
            
            # Split data
            train_size = int(len(df_prophet) * 0.8)
            train = df_prophet[:train_size]
            test = df_prophet[train_size:]
            
            # Build Prophet model
            model = Prophet(daily_seasonality=True, yearly_seasonality=True)
            model.fit(train)
            
            # Forecast
            future = model.make_future_dataframe(periods=len(test), freq='H')
            forecast = model.predict(future)
            
            # Evaluate on test set
            test_forecast = forecast.iloc[train_size:]['yhat'].values
            mse = mean_squared_error(test['y'], test_forecast)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test['y'], test_forecast)
            r2 = r2_score(test['y'], test_forecast)
            
            self.models['prophet'] = model
            self.predictions['prophet'] = {
                'train': train,
                'test': test,
                'forecast': forecast,
                'metrics': {'RMSE': rmse, 'MAE': mae, 'R2': r2}
            }
            
            print(f"  ✅ Prophet - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            return model, {'RMSE': rmse, 'MAE': mae, 'R2': r2}
            
        except Exception as e:
            print(f"  ⚠️ Prophet failed: {str(e)}")
            return None, None
    
    def build_lstm_model(self, target_col='temperature_celsius', lookback=24):
        """Build LSTM neural network model"""
        print("\n🤖 Building LSTM model...")
        
        try:
            df_ts = self.prepare_timeseries_data(target_col)
            data = df_ts[target_col].values.reshape(-1, 1)
            
            # Normalize
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)
            
            # Create sequences
            X, y = [], []
            for i in range(lookback, len(data_scaled)):
                X.append(data_scaled[i-lookback:i, 0])
                y.append(data_scaled[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Split
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Build LSTM
            model = Sequential([
                LSTM(50, activation='relu', return_sequences=True, input_shape=(lookback, 1)),
                Dropout(0.2),
                LSTM(50, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0, validation_split=0.1)
            
            # Predict
            y_pred = model.predict(X_test, verbose=0)
            
            # Inverse transform
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
            y_pred_inv = scaler.inverse_transform(y_pred)
            
            # Evaluate
            rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
            mae = mean_absolute_error(y_test_inv, y_pred_inv)
            r2 = r2_score(y_test_inv, y_pred_inv)
            
            self.models['lstm'] = model
            self.predictions['lstm'] = {
                'scaler': scaler,
                'y_test': y_test_inv,
                'y_pred': y_pred_inv,
                'metrics': {'RMSE': rmse, 'MAE': mae, 'R2': r2}
            }
            
            print(f"  ✅ LSTM - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            return model, {'RMSE': rmse, 'MAE': mae, 'R2': r2}
            
        except Exception as e:
            print(f"  ⚠️ LSTM failed: {str(e)}")
            return None, None
    
    def build_ml_models(self, target_col='temperature_celsius'):
        """Build Random Forest and XGBoost models"""
        print("\n🤖 Building ML models...")
        
        try:
            # Prepare features
            feature_cols = ['humidity', 'wind_kph', 'pressure_mb', 'cloud', 'month', 'hour']
            feature_cols = [col for col in feature_cols if col in self.df_cleaned.columns]
            
            if len(feature_cols) < 3:
                print("  ⚠️ Not enough features for ML models")
                return None
            
            X = self.df_cleaned[feature_cols].fillna(0)
            y = self.df_cleaned[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            results = {}
            
            # Random Forest
            print("  → Training Random Forest...")
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            
            rf_metrics = {
                'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
                'MAE': mean_absolute_error(y_test, rf_pred),
                'R2': r2_score(y_test, rf_pred)
            }
            
            self.models['random_forest'] = rf_model
            results['random_forest'] = rf_metrics
            print(f"  ✅ Random Forest - RMSE: {rf_metrics['RMSE']:.4f}, R²: {rf_metrics['R2']:.4f}")
            
            # XGBoost
            print("  → Training XGBoost...")
            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            
            xgb_metrics = {
                'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
                'MAE': mean_absolute_error(y_test, xgb_pred),
                'R2': r2_score(y_test, xgb_pred)
            }
            
            self.models['xgboost'] = xgb_model
            results['xgboost'] = xgb_metrics
            print(f"  ✅ XGBoost - RMSE: {xgb_metrics['RMSE']:.4f}, R²: {xgb_metrics['R2']:.4f}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.results['feature_importance'] = feature_importance
            self.predictions['ml_models'] = {
                'X_test': X_test,
                'y_test': y_test,
                'rf_pred': rf_pred,
                'xgb_pred': xgb_pred
            }
            
            return results
            
        except Exception as e:
            print(f"  ⚠️ ML models failed: {str(e)}")
            return None
    
    def create_ensemble(self):
        """Create ensemble of all models"""
        print("\n🎯 Creating ensemble model...")
        
        try:
            # Get predictions from all models
            predictions = []
            weights = []
            
            if 'ml_models' in self.predictions and self.predictions['ml_models'] is not None:
                rf_pred = self.predictions['ml_models']['rf_pred']
                xgb_pred = self.predictions['ml_models']['xgb_pred']
                y_test = self.predictions['ml_models']['y_test']
                
                predictions.extend([rf_pred, xgb_pred])
                weights.extend([0.5, 0.5])
                
                # Create ensemble prediction
                ensemble_pred = np.average(predictions, axis=0, weights=weights)
                
                # Evaluate ensemble
                rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
                mae = mean_absolute_error(y_test, ensemble_pred)
                r2 = r2_score(y_test, ensemble_pred)
                
                self.predictions['ensemble'] = {
                    'predictions': ensemble_pred,
                    'y_test': y_test,
                    'metrics': {'RMSE': rmse, 'MAE': mae, 'R2': r2}
                }
                
                print(f"  ✅ Ensemble - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                return {'RMSE': rmse, 'MAE': mae, 'R2': r2}
            else:
                print("  ⚠️ Not enough models for ensemble")
                return None
                
        except Exception as e:
            print(f"  ⚠️ Ensemble failed: {str(e)}")
            return None
    
    # ==================== ADVANCED ANALYSIS ====================
    
    def climate_analysis(self):
        """Analyze long-term climate patterns"""
        print("\n🌍 Performing climate analysis...")
        
        if self.df_cleaned is None:
            self.clean_data()
        
        analysis = {}
        
        # Temperature trends by month
        if 'month' in self.df_cleaned.columns and 'temperature_celsius' in self.df_cleaned.columns:
            monthly_temp = self.df_cleaned.groupby('month')['temperature_celsius'].agg(['mean', 'std', 'min', 'max'])
            analysis['monthly_patterns'] = monthly_temp
        
        # Seasonal analysis
        if 'month' in self.df_cleaned.columns:
            def get_season(month):
                if month in [12, 1, 2]:
                    return 'Winter'
                elif month in [3, 4, 5]:
                    return 'Spring'
                elif month in [6, 7, 8]:
                    return 'Summer'
                else:
                    return 'Fall'
            
            self.df_cleaned['season'] = self.df_cleaned['month'].apply(get_season)
            seasonal_stats = self.df_cleaned.groupby('season')['temperature_celsius'].describe()
            analysis['seasonal_patterns'] = seasonal_stats
        
        # Regional climate
        if 'country' in self.df_cleaned.columns:
            top_countries = self.df_cleaned['country'].value_counts().head(15).index
            country_climate = self.df_cleaned[self.df_cleaned['country'].isin(top_countries)].groupby('country')['temperature_celsius'].agg(['mean', 'std'])
            analysis['country_climate'] = country_climate
        
        self.results['climate_analysis'] = analysis
        print("✅ Climate analysis completed!")
        return analysis
    
    # def air_quality_analysis(self):
    #     """Analyze air quality correlation with weather"""
    #     print("\n💨 Analyzing air quality...")
        
    #     if self.df_cleaned is None:
    #         self.clean_data()
        
    #     # Look for air quality columns - only numeric ones
    #     potential_aq_cols = [col for col in self.df_cleaned.columns if any(x in col.lower() for x in ['pm', 'aqi', 'co', 'no2', 'o3', 'so2', 'pollutant'])]
        
    #     # Filter to only numeric columns
    #     aq_cols = [col for col in potential_aq_cols if pd.api.types.is_numeric_dtype(self.df_cleaned[col])]
        
    #     if not aq_cols:
    #         print("  ⚠️ No air quality columns found")
    #         return None
        
    #     analysis = {}
    #     weather_cols = ['temperature_celsius', 'humidity', 'wind_kph', 'pressure_mb', 'cloud', 'wind_mph']
    #     weather_cols = [col for col in weather_cols if col in self.df_cleaned.columns and pd.api.types.is_numeric_dtype(self.df_cleaned[col])]
        
    #     if not weather_cols:
    #         print("  ⚠️ No numeric weather columns found")
    #         return None
        
    #     for aq_col in aq_cols:
    #         correlations = {}
    #         for weather_col in weather_cols:
    #             try:
    #                 # Only calculate correlation if both columns have valid numeric data
    #                 valid_data = self.df_cleaned[[aq_col, weather_col]].dropna()
    #                 if len(valid_data) > 10:  # Need at least 10 data points
    #                     corr = valid_data.corr().iloc[0, 1]
    #                     if not np.isnan(corr):
    #                         correlations[weather_col] = corr
    #             except Exception as e:
    #                 continue
            
    #         if correlations:  # Only add if we found valid correlations
    #             analysis[aq_col] = correlations
        
    #     if not analysis:
    #         print("  ⚠️ No valid air quality correlations found")
    #         self.results['air_quality'] = None
    #         return None
        
    #     self.results['air_quality'] = analysis
    #     print("✅ Air quality analysis completed!")
    #     return analysis
    
    def spatial_analysis(self):
        """Analyze geographical patterns"""
        print("\n🗺️ Performing spatial analysis...")
        
        if self.df_cleaned is None:
            self.clean_data()
        
        analysis = {}
        
        # Country-wise statistics
        if 'country' in self.df_cleaned.columns:
            country_stats = self.df_cleaned.groupby('country').agg({
                'temperature_celsius': ['mean', 'std'],
                'humidity': 'mean',
                'wind_kph': 'mean'
            }).round(2)
            analysis['country_stats'] = country_stats
        
        # Latitude/longitude patterns if available
        if 'latitude' in self.df_cleaned.columns and 'temperature_celsius' in self.df_cleaned.columns:
            # Temperature by latitude bands
            self.df_cleaned['lat_band'] = pd.cut(self.df_cleaned['latitude'], bins=10)
            lat_temp = self.df_cleaned.groupby('lat_band')['temperature_celsius'].mean()
            analysis['latitude_temperature'] = lat_temp
        
        self.results['spatial_analysis'] = analysis
        print("✅ Spatial analysis completed!")
        return analysis
    
    # ==================== RUN COMPLETE PIPELINE ====================
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n" + "="*60)
        print("🚀 STARTING COMPLETE WEATHER ANALYSIS PIPELINE")
        print("="*60)
        
        # Step 1: Load and clean data
        self.clean_data()
        
        # Step 2: EDA
        self.generate_eda_insights()
        self.create_visualizations()
        
        # Step 3: Build forecasting models
        self.build_arima_model()
        self.build_prophet_model()
        self.build_ml_models()
        self.build_lstm_model()
        self.create_ensemble()
        
        # Step 4: Advanced analysis
        self.climate_analysis()
        # self.air_quality_analysis()
        self.spatial_analysis()

        # Step 5: Compile results
        model_comparison = self._compile_model_results()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
        
        return self.results, model_comparison
    
    def _compile_model_results(self):
        """Compile all model results for comparison"""
        comparison = {}
        
        for model_name in ['arima', 'prophet', 'lstm', 'random_forest', 'xgboost', 'ensemble']:
            if model_name in self.predictions and self.predictions[model_name] is not None:
                if 'metrics' in self.predictions[model_name]:
                    comparison[model_name] = self.predictions[model_name]['metrics']
        
        if comparison:
            comparison_df = pd.DataFrame(comparison).T
            comparison_df = comparison_df.sort_values('RMSE')
            self.results['model_comparison'] = comparison_df
            
            print("\n📊 Model Performance Comparison:")
            print(comparison_df.to_string())
        
        return comparison
    
    def save_results(self, output_dir='outputs'):
        """Save all results"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save cleaned data
        if self.df_cleaned is not None:
            self.df_cleaned.to_csv(f'{output_dir}/cleaned_data.csv', index=False)
        
        # Save models
        for name, model in self.models.items():
            if model is not None and name not in ['lstm']:
                joblib.dump(model, f'{output_dir}/{name}_model.pkl')
        
        print(f"\n💾 Results saved to {output_dir}/")


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = WeatherAnalysisPipeline('data/Global Weather Repository.csv')
    
    # Run complete analysis
    results, model_comparison = pipeline.run_complete_analysis()
    
    # Save results
    pipeline.save_results()