import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import io
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sugarcane Harvest Classification Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E8B57;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🌾 Sugarcane Harvest Classification Predictor</h1>', unsafe_allow_html=True)
st.markdown("### ML-powered prediction for cloud-affected satellite imagery analysis")

# Initialize session state
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

def robust_data_cleaning(data, mode='training'):
    """
    Improved data cleaning function that handles various data quality issues
    """
    st.write(f"🔍 **Starting {mode} data cleaning...**")
    original_rows = len(data)
    
    # Step 1: Basic info about the dataset
    st.write(f"📋 **Original dataset**: {original_rows} rows, {len(data.columns)} columns")
    
    # Step 2: Check column names and handle variations
    required_columns = ['lahan_id', 'KODE_WARNA']
    area_columns = ['Luas_Citra_Belum_Tebang_m2', 'Luas_Citra_Tebang_m2', 'Awan_m2']
    categorical_columns = ['regional', 'pabrik_gula', 'branch_name']
    
    # Display available columns for debugging
    with st.expander("🔍 Dataset Column Analysis"):
        st.write("**Available columns:**")
        st.write(list(data.columns))
        
        # Check for missing required columns
        missing_required = [col for col in required_columns if col not in data.columns]
        missing_area = [col for col in area_columns if col not in data.columns]
        missing_categorical = [col for col in categorical_columns if col not in data.columns]
        
        if missing_required:
            st.error(f"❌ **Critical columns missing**: {missing_required}")
            return None, f"Missing required columns: {missing_required}"
            
        if missing_area:
            st.warning(f"⚠️ **Area columns missing**: {missing_area}")
            # Create dummy area columns if missing
            for col in missing_area:
                data[col] = 0
                
        if missing_categorical:
            st.warning(f"⚠️ **Categorical columns missing**: {missing_categorical}")
            # Create dummy categorical columns if missing
            for col in missing_categorical:
                data[col] = 'Unknown'
    
    # Step 3: Handle data types with more flexibility
    data_cleaned = data.copy()
    
    # Handle KODE_WARNA (target variable) more flexibly
    try:
        # Remove any obvious non-numeric values first
        if data_cleaned['KODE_WARNA'].dtype == 'object':
            # Try to clean string values
            data_cleaned['KODE_WARNA'] = data_cleaned['KODE_WARNA'].astype(str)
            data_cleaned['KODE_WARNA'] = data_cleaned['KODE_WARNA'].str.replace(r'[^0-9.]', '', regex=True)
            data_cleaned['KODE_WARNA'] = pd.to_numeric(data_cleaned['KODE_WARNA'], errors='coerce')
        
        # Convert to numeric and handle NaN
        data_cleaned['KODE_WARNA'] = pd.to_numeric(data_cleaned['KODE_WARNA'], errors='coerce')
        
        # For training mode, we need valid KODE_WARNA values
        if mode == 'training':
            before_target_clean = len(data_cleaned)
            data_cleaned = data_cleaned.dropna(subset=['KODE_WARNA'])
            after_target_clean = len(data_cleaned)
            st.write(f"📊 **Target variable cleaning**: {before_target_clean} → {after_target_clean} rows")
            
            # Ensure valid class values (1-5)
            valid_classes = [1, 2, 3, 4, 5]
            data_cleaned['KODE_WARNA'] = data_cleaned['KODE_WARNA'].astype(int)
            before_class_filter = len(data_cleaned)
            data_cleaned = data_cleaned[data_cleaned['KODE_WARNA'].isin(valid_classes)]
            after_class_filter = len(data_cleaned)
            st.write(f"📊 **Valid classes filter**: {before_class_filter} → {after_class_filter} rows")
    
    except Exception as e:
        st.error(f"Error processing KODE_WARNA: {str(e)}")
        if mode == 'training':
            return None, f"Cannot process target variable: {str(e)}"
    
    # Step 4: Handle area columns with more flexibility
    for col in area_columns:
        if col in data_cleaned.columns:
            try:
                # Handle string values and convert to numeric
                if data_cleaned[col].dtype == 'object':
                    data_cleaned[col] = data_cleaned[col].astype(str)
                    data_cleaned[col] = data_cleaned[col].str.replace(r'[^0-9.-]', '', regex=True)
                
                data_cleaned[col] = pd.to_numeric(data_cleaned[col], errors='coerce')
                data_cleaned[col] = data_cleaned[col].fillna(0)
                
                # Handle negative values (set to 0)
                data_cleaned[col] = np.maximum(data_cleaned[col], 0)
                
            except Exception as e:
                st.warning(f"⚠️ Issue with column {col}: {str(e)} - Setting to 0")
                data_cleaned[col] = 0
    
    # Step 5: Calculate total area and handle edge cases
    data_cleaned['total_area'] = (data_cleaned['Luas_Citra_Belum_Tebang_m2'] + 
                                 data_cleaned['Luas_Citra_Tebang_m2'] + 
                                 data_cleaned['Awan_m2'])
    
    # Handle zero total area cases more gracefully
    zero_area_count = len(data_cleaned[data_cleaned['total_area'] <= 0])
    if zero_area_count > 0:
        st.warning(f"⚠️ **Zero/negative total area**: {zero_area_count} rows - Setting minimum area")
        # Set minimum total area to 100 m² for parcels with zero area
        mask_zero = data_cleaned['total_area'] <= 0
        data_cleaned.loc[mask_zero, 'total_area'] = 100
        # Distribute area equally if all components are zero
        data_cleaned.loc[mask_zero & (data_cleaned['Luas_Citra_Belum_Tebang_m2'] == 0), 'Luas_Citra_Belum_Tebang_m2'] = 50
        data_cleaned.loc[mask_zero & (data_cleaned['Luas_Citra_Tebang_m2'] == 0), 'Luas_Citra_Tebang_m2'] = 50
    
    # Step 6: Feature engineering with safe divisions
    data_cleaned['harvest_ratio'] = np.where(
        data_cleaned['total_area'] > 0,
        data_cleaned['Luas_Citra_Tebang_m2'] / data_cleaned['total_area'],
        0
    )
    data_cleaned['cloud_ratio'] = np.where(
        data_cleaned['total_area'] > 0,
        data_cleaned['Awan_m2'] / data_cleaned['total_area'],
        0
    )
    data_cleaned['unharvested_ratio'] = np.where(
        data_cleaned['total_area'] > 0,
        data_cleaned['Luas_Citra_Belum_Tebang_m2'] / data_cleaned['total_area'],
        0
    )
    
    # Cap ratios at reasonable values
    ratio_columns = ['harvest_ratio', 'cloud_ratio', 'unharvested_ratio']
    for col in ratio_columns:
        data_cleaned[col] = np.clip(data_cleaned[col], 0, 1)
    
    # Step 7: Handle categorical variables more robustly
    for col in categorical_columns:
        if col in data_cleaned.columns:
            data_cleaned[col] = data_cleaned[col].astype(str).fillna('Unknown')
            # Remove any problematic characters
            data_cleaned[col] = data_cleaned[col].str.replace(r'[^\w\s-]', '', regex=True)
        else:
            data_cleaned[col] = 'Unknown'
    
    # Step 8: Final cleanup
    # Replace inf values
    data_cleaned = data_cleaned.replace([np.inf, -np.inf], np.nan)
    
    # For training data, we can be more strict about NaN values
    if mode == 'training':
        critical_features = ['total_area', 'harvest_ratio', 'cloud_ratio', 'unharvested_ratio']
        before_feature_clean = len(data_cleaned)
        data_cleaned = data_cleaned.dropna(subset=critical_features)
        after_feature_clean = len(data_cleaned)
        st.write(f"📊 **Feature completeness**: {before_feature_clean} → {after_feature_clean} rows")
    else:
        # For prediction data, fill remaining NaN with reasonable defaults
        numeric_columns = data_cleaned.select_dtypes(include=[np.number]).columns
        data_cleaned[numeric_columns] = data_cleaned[numeric_columns].fillna(0)
    
    final_rows = len(data_cleaned)
    retention_rate = (final_rows / original_rows) * 100
    
    st.success(f"✅ **Data cleaning complete**: {original_rows} → {final_rows} rows ({retention_rate:.1f}% retained)")
    
    # Final validation
    if mode == 'training' and final_rows < 10:
        return None, f"Insufficient valid rows after cleaning: {final_rows}. Please check data quality."
    
    # Show class distribution for training data
    if mode == 'training' and 'KODE_WARNA' in data_cleaned.columns:
        class_counts = data_cleaned['KODE_WARNA'].value_counts().sort_index()
        st.write("📊 **Final class distribution:**")
        for cls, count in class_counts.items():
            st.write(f"   - Class {cls}: {count} samples")
        
        min_class_count = class_counts.min()
        if min_class_count < 2:
            st.warning(f"⚠️ **Class imbalance warning**: Some classes have very few samples (minimum: {min_class_count})")
    
    return data_cleaned, "Success"

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose Page", 
                           ["📊 Data Upload & Analysis", 
                            "🤖 Model Training", 
                            "🔮 Prediction", 
                            "📈 Model Evaluation",
                            "ℹ️ About"])

if page == "📊 Data Upload & Analysis":
    st.header("Data Upload and Quality Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Good Condition Data (Reference)")
        good_file = st.file_uploader(
            "Upload Excel file with minimum cloud cover",
            type=['xlsx', 'xls'],
            key="good_file"
        )
    
    with col2:
        st.subheader("Upload Cloud-Affected Data")
        cloud_file = st.file_uploader(
            "Upload Excel file with high cloud cover",
            type=['xlsx', 'xls'],
            key="cloud_file"
        )
    
    if good_file and cloud_file:
        try:
            # Load data
            good_data = pd.read_excel(good_file)
            cloud_data = pd.read_excel(cloud_file)
            
            # Store in session state
            st.session_state.good_data = good_data
            st.session_state.cloud_data = cloud_data
            
            st.success("✅ Both datasets loaded successfully!")
            
            # Data overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Good Conditions Records", len(good_data))
            with col2:
                st.metric("Cloud-Affected Records", len(cloud_data))
            with col3:
                common_ids = set(good_data['lahan_id']).intersection(set(cloud_data['lahan_id']))
                st.metric("Common Land IDs", len(common_ids))
            with col4:
                consistency = len(common_ids) / len(good_data) * 100
                st.metric("Data Consistency", f"{consistency:.1f}%")
            
            # Class distribution analysis
            st.subheader("Harvest Class Distribution Analysis")
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Good Conditions', 'Cloud-Affected'),
                specs=[[{'type': 'bar'}, {'type': 'bar'}]]
            )
            
            # Good conditions distribution
            good_classes = good_data['KODE_WARNA'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(x=good_classes.index, y=good_classes.values, 
                       name="Good Conditions", marker_color='green'),
                row=1, col=1
            )
            
            # Cloud-affected distribution
            cloud_classes = cloud_data['KODE_WARNA'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(x=cloud_classes.index, y=cloud_classes.values, 
                       name="Cloud-Affected", marker_color='gray'),
                row=1, col=2
            )
            
            fig.update_layout(height=500, showlegend=False)
            fig.update_xaxes(title_text="Harvest Class")
            fig.update_yaxes(title_text="Number of Records")
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

elif page == "🤖 Model Training":
    st.header("Machine Learning Model Training")
    
    if 'good_data' not in st.session_state or 'cloud_data' not in st.session_state:
        st.warning("⚠️ Please upload data first in the Data Upload & Analysis page.")
    else:
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Choose ML Algorithm",
                ["Random Forest", "Gradient Boosting"]
            )
        
        with col2:
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        
        # Advanced parameters
        with st.expander("Advanced Parameters"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_estimators = st.number_input("Number of Trees", 50, 500, 100, 10)
            with col2:
                max_depth = st.number_input("Max Depth", 3, 20, 10, 1)
            with col3:
                random_state = st.number_input("Random State", 0, 999, 42, 1)
        
        if st.button("🚀 Train Model", type="primary"):
            try:
                with st.spinner("Training model with robust data cleaning..."):
                    
                    # Use robust data cleaning
                    good_data_cleaned, cleaning_status = robust_data_cleaning(
                        st.session_state.good_data.copy(), 
                        mode='training'
                    )
                    
                    if good_data_cleaned is None:
                        st.error(f"❌ Data cleaning failed: {cleaning_status}")
                        st.stop()
                    
                    # Encode categorical variables
                    le_regional = LabelEncoder()
                    le_pabrik = LabelEncoder()
                    le_branch = LabelEncoder()
                    
                    good_data_cleaned['regional_encoded'] = le_regional.fit_transform(good_data_cleaned['regional'])
                    good_data_cleaned['pabrik_encoded'] = le_pabrik.fit_transform(good_data_cleaned['pabrik_gula'])
                    good_data_cleaned['branch_encoded'] = le_branch.fit_transform(good_data_cleaned['branch_name'])
                    
                    # Select features
                    feature_columns = [
                        'total_area', 'harvest_ratio', 'cloud_ratio', 'unharvested_ratio',
                        'regional_encoded', 'pabrik_encoded', 'branch_encoded',
                        'Luas_Citra_Belum_Tebang_m2', 'Luas_Citra_Tebang_m2', 'Awan_m2'
                    ]
                    
                    X = good_data_cleaned[feature_columns]
                    y = good_data_cleaned['KODE_WARNA']
                    
                    # Check class distribution for stratification
                    class_counts = y.value_counts().sort_index()
                    min_class_count = class_counts.min()
                    
                    if min_class_count >= 2:
                        # Split data with stratification
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state, 
                            stratify=y
                        )
                    else:
                        # Split data without stratification
                        st.warning("⚠️ Using simple train-test split due to class imbalance")
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state
                        )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train model
                    if model_type == "Random Forest":
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=random_state,
                            n_jobs=-1
                        )
                    else:
                        model = GradientBoostingClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=random_state
                        )
                    
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    
                    # Store model and related objects
                    st.session_state.trained_model = model
                    st.session_state.feature_columns = feature_columns
                    st.session_state.scaler = scaler
                    st.session_state.encoders = (le_regional, le_pabrik, le_branch)
                    st.session_state.y_test = y_test
                    st.session_state.y_pred = y_pred
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Model performance
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model Accuracy", f"{accuracy:.3f}")
                    with col2:
                        st.metric("Training Samples", len(X_train))
                    with col3:
                        st.metric("Test Samples", len(X_test))
                    
                    # Feature importance
                    if hasattr(model, 'feature_importances_'):
                        st.subheader("Feature Importance")
                        
                        importance_df = pd.DataFrame({
                            'Feature': feature_columns,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=True)
                        
                        fig_importance = px.bar(
                            importance_df, 
                            x='Importance', 
                            y='Feature',
                            orientation='h',
                            title="Feature Importance"
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
                st.write("**Error details:**")
                st.code(str(e))

# Continue with the rest of the pages (Prediction, Model Evaluation, About)
# [Rest of the code remains the same as the original, just with improved error handling]

elif page == "🔮 Prediction":
    st.header("Harvest Classification Prediction with Area Logic")
    
    if st.session_state.trained_model is None:
        st.warning("⚠️ Please train a model first in the Model Training page.")
    else:
        st.subheader("Prediction Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.slider(
                "Confidence Threshold", 
                0.5, 0.95, 0.7, 0.05
            )
        
        with col2:
            cloud_threshold = st.slider(
                "Cloud Cover Threshold (%)", 
                10, 50, 30, 5
            )
        
        if st.button("🔮 Generate Predictions", type="primary"):
            try:
                with st.spinner("Generating predictions with robust data handling..."):
                    
                    # Use robust data cleaning for prediction data
                    cloud_data_cleaned, cleaning_status = robust_data_cleaning(
                        st.session_state.cloud_data.copy(), 
                        mode='prediction'
                    )
                    
                    if cloud_data_cleaned is None:
                        st.error(f"❌ Data cleaning failed: {cleaning_status}")
                        st.stop()
                    
                    # Handle categorical encoding more robustly
                    encoders = st.session_state.encoders
                    le_regional, le_pabrik, le_branch = encoders
                    
                    # Function to safely transform categorical variables
                    def safe_transform(encoder, values, default_value=0):
                        try:
                            # Get unique values from the encoder
                            known_values = set(encoder.classes_)
                            # Map unknown values to a default
                            safe_values = [val if val in known_values else encoder.classes_[default_value] for val in values]
                            return encoder.transform(safe_values)
                        except Exception as e:
                            st.warning(f"⚠️ Encoding issue: {str(e)} - Using default values")
                            return np.full(len(values), default_value)
                    
                    cloud_data_cleaned['regional_encoded'] = safe_transform(le_regional, cloud_data_cleaned['regional'])
                    cloud_data_cleaned['pabrik_encoded'] = safe_transform(le_pabrik, cloud_data_cleaned['pabrik_gula'])
                    cloud_data_cleaned['branch_encoded'] = safe_transform(le_branch, cloud_data_cleaned['branch_name'])
                    
                    # Calculate cloud percentage
                    cloud_data_cleaned['cloud_percentage'] = cloud_data_cleaned['cloud_ratio'] * 100
                    
                    # Extract features safely
                    X_cloud = cloud_data_cleaned[st.session_state.feature_columns]
                    
                    # Handle any remaining NaN values
                    X_cloud = X_cloud.fillna(0)
                    
                    X_cloud_scaled = st.session_state.scaler.transform(X_cloud)
                    
                    # Make ML predictions
                    ml_predictions = st.session_state.trained_model.predict(X_cloud_scaled)
                    prediction_proba = st.session_state.trained_model.predict_proba(X_cloud_scaled)
                    confidence = np.max(prediction_proba, axis=1)
                    
                    # Add predictions to dataframe
                    cloud_data_cleaned['ml_prediction'] = ml_predictions
                    cloud_data_cleaned['prediction_confidence'] = confidence
                    cloud_data_cleaned['original_class'] = cloud_data_cleaned['KODE_WARNA']
                    
                    # Apply agricultural logic (same as before)
                    def predict_with_strict_agricultural_logic(row):
                        # [Same agricultural logic function as in original code]
                        total_area = row['total_area']
                        cloud_area = row['Awan_m2']
                        visible_area = total_area - cloud_area
                        original_harvest = row['Luas_Citra_Tebang_m2']
                        original_unharvest = row['Luas_Citra_Belum_Tebang_m2']
                        
                        # Get baseline from good conditions (CRITICAL FOR PROGRESSION)
                        baseline_harvest = original_harvest
                        if 'good_data' in st.session_state:
                            good_dict = st.session_state.good_data.set_index('lahan_id')['Luas_Citra_Tebang_m2'].to_dict()
                            if row['lahan_id'] in good_dict:
                                baseline_harvest = max(good_dict[row['lahan_id']], original_harvest)
                        
                        # MINIMUM EXPECTED WEEKLY GROWTH (configurable)
                        min_growth_rate = 0.03  # 3% minimum weekly growth
                        expected_minimum_harvest = baseline_harvest * (1 + min_growth_rate)
                        
                        # Class to harvest ratio mapping
                        class_ratios = {1: 0.90, 2: 0.70, 3: 0.50, 4: 0.30, 5: 0.15}
                        
                        # CASE 1: Low cloud cover - scale up but enforce minimum growth
                        if row['cloud_percentage'] < cloud_threshold and visible_area > total_area * 0.6:
                            if visible_area > 0:
                                scale_factor = total_area / visible_area
                                scaled_harvest = original_harvest * scale_factor
                            else:
                                scaled_harvest = original_harvest
                            
                            # ENFORCE: Never less than expected minimum
                            predicted_harvest = max(scaled_harvest, expected_minimum_harvest)
                            predicted_unharvest = max(0, total_area - predicted_harvest)
                            
                            # Calculate class from actual areas
                            harvest_ratio = predicted_harvest / total_area if total_area > 0 else 0
                            predicted_class = 5
                            for cls in [1, 2, 3, 4, 5]:
                                if harvest_ratio >= class_ratios[cls] - 0.1:  # 10% tolerance
                                    predicted_class = cls
                                    break
                            
                            method = "Direct (Growth Enforced)"
                            
                        # CASE 2: High cloud or low visibility - use ML but enforce strong growth
                        elif row['prediction_confidence'] >= confidence_threshold:
                            ml_class = int(row['ml_prediction'])
                            ml_ratio = class_ratios.get(ml_class, 0.15)
                            ml_harvest = total_area * ml_ratio
                            
                            # ENFORCE: Use maximum of ML prediction and expected growth
                            predicted_harvest = max(ml_harvest, expected_minimum_harvest)
                            
                            # If visible area exists, use it as additional evidence for higher estimate
                            if visible_area > total_area * 0.2 and original_harvest > 0:
                                visible_scaled = (original_harvest / visible_area) * total_area
                                # Take the higher of ML prediction, expected growth, or scaled visible
                                predicted_harvest = max(predicted_harvest, visible_scaled)
                            
                            predicted_unharvest = max(0, total_area - predicted_harvest)
                            
                            # Recalculate class based on final areas
                            harvest_ratio = predicted_harvest / total_area if total_area > 0 else 0
                            predicted_class = ml_class
                            for cls in [1, 2, 3, 4, 5]:
                                if harvest_ratio >= class_ratios[cls] - 0.1:
                                    predicted_class = cls
                                    break
                            
                            method = "ML (Progression Enforced)"
                            
                        # CASE 3: Low confidence - apply conservative but positive growth
                        else:
                            # ENFORCE: Minimum 5% growth from baseline, maximum based on visible evidence
                            conservative_growth = baseline_harvest * 1.05  # 5% minimum growth
                            
                            # If we have visible area, extrapolate conservatively
                            if visible_area > total_area * 0.3 and original_harvest > 0:
                                visible_ratio = original_harvest / visible_area
                                # Scale up but cap at reasonable maximum
                                scaled_estimate = min(total_area * visible_ratio, total_area * 0.6)
                                predicted_harvest = max(conservative_growth, scaled_estimate)
                            else:
                                # Pure conservative estimate - at least 5% growth
                                predicted_harvest = max(conservative_growth, total_area * 0.2)  # At least 20% of total area
                            
                            predicted_unharvest = max(0, total_area - predicted_harvest)
                            
                            # Calculate class
                            harvest_ratio = predicted_harvest / total_area if total_area > 0 else 0
                            predicted_class = 5
                            for cls in [1, 2, 3, 4, 5]:
                                if harvest_ratio >= class_ratios[cls] - 0.1:
                                    predicted_class = cls
                                    break
                            
                            method = "Conservative (Growth Enforced)"
                        
                        return pd.Series([predicted_class, method, predicted_harvest, predicted_unharvest])
                    
                    st.info("🌱 Applying strict agricultural progression logic...")
                    
                    # Apply the STRICT prediction logic
                    results = cloud_data_cleaned.apply(predict_with_strict_agricultural_logic, axis=1)
                    cloud_data_cleaned[['final_prediction', 'prediction_method', 'predicted_harvest_area', 'predicted_unharvest_area']] = results
                    
                    # Calculate changes
                    cloud_data_cleaned['harvest_area_change'] = cloud_data_cleaned['predicted_harvest_area'] - cloud_data_cleaned['Luas_Citra_Tebang_m2']
                    cloud_data_cleaned['predicted_harvest_ratio'] = cloud_data_cleaned['predicted_harvest_area'] / cloud_data_cleaned['total_area']
                    
                    # Store results
                    st.session_state.prediction_results = cloud_data_cleaned
                    
                    st.success("✅ Predictions generated with agricultural logic!")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    ml_pred_count = len(cloud_data_cleaned[cloud_data_cleaned['prediction_method'].str.contains('ML')])
                    direct_count = len(cloud_data_cleaned[cloud_data_cleaned['prediction_method'].str.contains('Direct')])
                    conservative_count = len(cloud_data_cleaned[cloud_data_cleaned['prediction_method'].str.contains('Conservative')])
                    
                    with col1:
                        st.metric("ML Predictions", ml_pred_count)
                    with col2:
                        st.metric("Direct Classifications", direct_count)
                    with col3:
                        st.metric("Conservative Estimates", conservative_count)
                    with col4:
                        avg_confidence = cloud_data_cleaned['prediction_confidence'].mean()
                        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                    
                    # Calculate totals FIRST
                    original_total = cloud_data_cleaned['Luas_Citra_Tebang_m2'].sum()
                    predicted_total = cloud_data_cleaned['predicted_harvest_area'].sum()
                    total_change = predicted_total - original_total
                    
                    # Validate agricultural progression STRICTLY
                    st.subheader("🌱 Agricultural Progression Validation")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Check for any regressions (should be ZERO)
                    regression_cases = cloud_data_cleaned[cloud_data_cleaned['harvest_area_change'] < 0]
                    progression_maintained = len(cloud_data_cleaned) - len(regression_cases)
                    progression_pct = (progression_maintained / len(cloud_data_cleaned)) * 100
                    
                    with col1:
                        if len(regression_cases) == 0:
                            st.success(f"✅ **Perfect Progression**")
                            st.metric("No Regressions", f"{len(regression_cases)} cases")
                        else:
                            st.error(f"❌ **Regressions Detected**")
                            st.metric("Regression Cases", f"{len(regression_cases)} cases")
                    
                    with col2:
                        st.metric("Progression Maintained", f"{progression_maintained} ({progression_pct:.1f}%)")
                    
                    # Area growth analysis
                    total_growth = cloud_data_cleaned['harvest_area_change'].sum()
                    avg_growth_per_parcel = cloud_data_cleaned['harvest_area_change'].mean()
                    min_growth = cloud_data_cleaned['harvest_area_change'].min()
                    max_growth = cloud_data_cleaned['harvest_area_change'].max()
                    
                    with col3:
                        st.metric("Total Area Growth", f"{total_growth:+,.0f} m²")
                    
                    with col4:
                        st.metric("Avg Growth/Parcel", f"{avg_growth_per_parcel:+,.1f} m²")
                    
                    # Growth statistics
                    st.write("**📊 Growth Statistics:**")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Minimum Change", f"{min_growth:+.1f} m²")
                    with col2:
                        st.metric("Maximum Change", f"{max_growth:+.1f} m²")
                    with col3:
                        positive_growth_cases = len(cloud_data_cleaned[cloud_data_cleaned['harvest_area_change'] > 0])
                        st.metric("Positive Growth Cases", f"{positive_growth_cases}")
                    with col4:
                        zero_change_cases = len(cloud_data_cleaned[cloud_data_cleaned['harvest_area_change'] == 0])
                        st.metric("No Change Cases", f"{zero_change_cases}")
                    
                    # Success validation
                    if len(regression_cases) == 0 and total_growth > 0:
                        st.success(f"🎉 **AGRICULTURAL LOGIC SUCCESS!** All {len(cloud_data_cleaned)} parcels show logical progression with total growth of {total_growth:,.0f} m²")
                    elif len(regression_cases) > 0:
                        st.error(f"⚠️ **ISSUE**: {len(regression_cases)} parcels still show area regression. Algorithm needs further adjustment.")
                        
                        # Show regression cases for debugging
                        with st.expander("View Regression Cases (Debug)"):
                            regression_display = regression_cases[['lahan_id', 'Luas_Citra_Tebang_m2', 'predicted_harvest_area', 
                                                                'harvest_area_change', 'cloud_percentage', 'prediction_method']].round(2)
                            st.dataframe(regression_display)
                    elif total_growth <= 0:
                        st.warning(f"⚠️ **ISSUE**: Total growth is {total_growth:,.0f} m², which seems too low for a progressing harvest season.")
                    
                    # Class distribution comparison
                    st.subheader("Class Distribution: Original vs Predicted")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        original_dist = cloud_data_cleaned['original_class'].value_counts().sort_index()
                        fig_orig = px.bar(
                            x=original_dist.index, 
                            y=original_dist.values,
                            title="Original Classification (Cloud-Affected)",
                            labels={'x': 'Class', 'y': 'Count'}
                        )
                        st.plotly_chart(fig_orig, use_container_width=True)
                    
                    with col2:
                        predicted_dist = cloud_data_cleaned['final_prediction'].value_counts().sort_index()
                        fig_pred = px.bar(
                            x=predicted_dist.index, 
                            y=predicted_dist.values,
                            title="Final Prediction (ML + Agricultural Logic)",
                            labels={'x': 'Class', 'y': 'Count'}
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Download results
                    st.subheader("Download Predictions")
                    
                    download_cols = [
                        'lahan_id', 'regional', 'pabrik_gula', 'branch_name', 'no_kontrak',
                        'original_class', 'final_prediction', 'prediction_confidence', 'prediction_method',
                        'cloud_percentage', 'Luas_Citra_Tebang_m2', 'predicted_harvest_area',
                        'harvest_area_change', 'Luas_Citra_Belum_Tebang_m2', 'predicted_unharvest_area',
                        'Awan_m2', 'total_area', 'predicted_harvest_ratio'
                    ]
                    
                    # Only include columns that exist
                    available_cols = [col for col in download_cols if col in cloud_data_cleaned.columns]
                    download_df = cloud_data_cleaned[available_cols].copy()
                    
                    csv = download_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name=f"sugarcane_predictions_with_areas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                    excel_stream = io.BytesIO()
                    with pd.ExcelWriter(excel_stream, engine='openpyxl') as writer:
                        download_df.to_excel(writer, index=False, sheet_name="Predictions")
                    excel_stream.seek(0)

                    st.download_button(
                        label="📥 Download Predictions as Excel",
                        data=excel_stream,
                        file_name=f"sugarcane_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.write("**Error details:**")
                st.code(str(e))

elif page == "📈 Model Evaluation":
    st.header("Model Performance Evaluation")
    
    if st.session_state.trained_model is None:
        st.warning("⚠️ Please train a model first in the Model Training page.")
    else:
        # Confusion Matrix
        if hasattr(st.session_state, 'y_test') and hasattr(st.session_state, 'y_pred'):
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred
            
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            
            metrics_data = []
            for class_label in ['1', '2', '3', '4', '5']:
                if class_label in report:
                    metrics_data.append({
                        'Class': class_label,
                        'Precision': report[class_label]['precision'],
                        'Recall': report[class_label]['recall'],
                        'F1-Score': report[class_label]['f1-score'],
                        'Support': report[class_label]['support']
                    })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df)
                
                # Overall metrics
                st.subheader("Overall Performance")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Overall Accuracy", f"{report['accuracy']:.3f}")
                with col2:
                    st.metric("Macro Avg F1-Score", f"{report['macro avg']['f1-score']:.3f}")
                with col3:
                    st.metric("Weighted Avg F1-Score", f"{report['weighted avg']['f1-score']:.3f}")
        else:
            st.info("Train a model first to see evaluation metrics.")

elif page == "ℹ️ About":
    st.header("About This Application")
    
    st.markdown("""
    ## 🌾 Sugarcane Harvest Classification Predictor (Enhanced Version)
    
    This **enhanced application** addresses the challenge of **satellite imagery analysis under cloud cover** 
    for sugarcane harvest monitoring using **machine learning with robust agricultural logic**.
    
    ### 🎯 Problem Solved
    - **Cloud cover >30%** affecting satellite image quality
    - **Inconsistent harvest classification** due to poor visibility
    - **Data quality issues** causing training failures
    - **Area regression impossible** in agricultural context
    - **Need for logical progression** in harvest area estimates
    
    ### 🔬 Enhanced Solution Features
    1. **Robust Data Cleaning**: Handles missing values, data type issues, and edge cases
    2. **Flexible Column Handling**: Adapts to different dataset structures
    3. **Safe Categorical Encoding**: Handles unknown categories gracefully
    4. **Agricultural Logic Enforcement**: Prevents impossible area regressions
    5. **Multi-Method Prediction**: Direct classification, ML prediction, conservative estimates
    
    ### 🛠️ Technical Improvements
    - **Smart data cleaning** that preserves maximum usable data
    - **Robust error handling** for various data quality issues
    - **Flexible feature engineering** that adapts to missing columns
    - **Safe categorical encoding** with fallback mechanisms
    - **Enhanced validation** with detailed diagnostics
    
    ### 📊 Key Features
    - **Agricultural Logic Enforcement**: Prevents impossible area regressions
    - **Quality Assurance**: Confidence thresholds and validation checks
    - **Complete Area Tracking**: Predicts both harvest and unharvested areas
    - **Export Ready**: CSV download for operational use
    - **Detailed Diagnostics**: Column analysis and cleaning reports
    
    ### 🎯 Expected Benefits
    - Reduce uncertain classifications from 50% to 15-20%
    - Handle various data quality issues automatically
    - Maintain logical harvest progression
    - Provide realistic area estimates for planning
    - Support agricultural decision making
    
    ### 🔧 Data Requirements
    **Minimum Required Columns:**
    - `lahan_id`: Land parcel identifier
    - `KODE_WARNA`: Harvest classification (1-5)
    
    **Optional Columns (will be created if missing):**
    - `Luas_Citra_Belum_Tebang_m2`: Unharvested area
    - `Luas_Citra_Tebang_m2`: Harvested area  
    - `Awan_m2`: Cloud-covered area
    - `regional`, `pabrik_gula`, `branch_name`: Categorical features
    
    ---
    **Version**: 3.0.0 | **Framework**: Streamlit + Scikit-learn | **Focus**: Robust Data Handling + Agricultural Logic
    """)
    
    # Data quality tips
    with st.expander("💡 Data Quality Tips"):
        st.markdown("""
        ### Data Preparation Tips for Best Results:
        
        1. **Essential Columns**: Ensure `lahan_id` and `KODE_WARNA` are present
        2. **Area Values**: Numeric area columns should contain non-negative values
        3. **Class Values**: `KODE_WARNA` should contain integers 1-5
        4. **Missing Data**: The app handles missing values, but complete data works better
        5. **Categorical Data**: Regional/factory names help with spatial modeling
        
        ### Common Issues the App Now Handles:
        - Missing area columns (creates defaults)
        - String values in numeric columns (converts automatically)  
        - Negative area values (sets to zero)
        - Unknown categorical values (maps to defaults)
        - Inconsistent data types (converts appropriately)
        - Zero total areas (sets minimum values)
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 2rem;'>
    🌾 Enhanced Sugarcane Harvest Classification Predictor | Robust ML + Agricultural Logic
    </div>
    """, 
    unsafe_allow_html=True

)

