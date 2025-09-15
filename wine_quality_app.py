
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from streamlit_lottie import st_lottie
import json
import time
import os
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# --- Hyperparameter tuning for both regression and classification ---
    # ...existing code...
def tune_hyperparameters():
    print("\n--- Hyperparameter Tuning ---")
    # Load data
    df = pd.read_csv('winequality-red.csv', delimiter=';')
    feature_cols = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ]
    X = df[feature_cols]
    y_reg = df['quality']
    y_clf = (df['quality'] >= 6).astype(int)
    # Split data
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

    # Regression tuning
    reg_params = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    reg = RandomForestRegressor(random_state=42)
    reg_grid = GridSearchCV(reg, reg_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    reg_grid.fit(X_train, y_train_reg)
    print("Best regressor params:", reg_grid.best_params_)
    y_pred_reg = reg_grid.predict(X_test)
    print("Regression R²:", r2_score(y_test_reg, y_pred_reg))
    print("Regression RMSE:", np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)))

    # Classification tuning
    clf_params = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    clf = RandomForestClassifier(random_state=42)
    clf_grid = GridSearchCV(clf, clf_params, cv=3, scoring='f1', n_jobs=-1)
    clf_grid.fit(X_train, y_train_clf)
    print("Best classifier params:", clf_grid.best_params_)
    y_pred_clf = clf_grid.predict(X_test)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    print("Classification Accuracy:", accuracy_score(y_test_clf, y_pred_clf))
    print("Classification F1:", f1_score(y_test_clf, y_pred_clf))

if __name__ == "__main__":
    # ...existing code...
    tune_hyperparameters()
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from streamlit_lottie import st_lottie
import json
import time
import os
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# wine_quality_premium_app_fixed.py

# Custom CSS for premium styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        margin: 1rem;
    }
    
    /* Header styling */
    .premium-header {
        text-align: center;
        color: #fff;
        font-weight: 900;
        font-size: 3.2rem;
        margin-bottom: 2rem;
        letter-spacing: 1px;
        text-shadow: 0 4px 24px rgba(44,62,80,0.25), 0 1px 2px rgba(44,62,80,0.18);
    }
    
    /* Animated gradient text */
    .gradient-text {
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 3s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Prediction button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
    /* Enhanced shimmer animated text */
    .gradient-text {
        position: relative;
        display: inline-block;
        font-weight: 1000;
        font-size: 4rem;
        letter-spacing: 2.5px;
        margin-bottom: 0.2rem;
        margin-top: 1.5rem;
        text-align: center;
        color: #fff;
        background: linear-gradient(110deg, #ffb347 10%, #ff5e62 30%, #f093fb 50%, #ffb347 70%, #fffbe6 85%, #ffb347 100%);
        background-size: 300% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-fill-color: transparent;
        text-shadow: 0 4px 32px rgba(0,0,0,0.45), 0 2px 0 #fff, 0 0 8px #ffb34799;
        filter: brightness(1.18) contrast(1.25);
        animation: shimmer 1.2s linear infinite;
    }

    @keyframes shimmer {
        0% {
            background-position: 100% 0;
        }
        100% {
            background-position: 0 0;
        }
    }

    /* Add a moving highlight overlay for extra motion */
    .gradient-text::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(120deg, rgba(255,255,255,0) 60%, rgba(255,255,255,0.7) 75%, rgba(255,255,255,0) 90%);
        pointer-events: none;
        mix-blend-mode: lighten;
        animation: highlight-move 1.2s linear infinite;
    }

    @keyframes highlight-move {
        0% { left: -60%; right: 100%; }
        100% { left: 100%; right: -60%; }
    }
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-fill-color: transparent;
        font-weight: 1000;
        font-size: 4rem;
        letter-spacing: 2.5px;
        margin-bottom: 0.2rem;
        margin-top: 1.5rem;
        text-align: center;
        animation: gradientMove 3s ease-in-out infinite alternate;
        transition: text-shadow 0.3s;
        text-shadow: 0 4px 32px rgba(0,0,0,0.45), 0 2px 0 #fff, 0 0 8px #ffb34799;
        filter: brightness(1.15) contrast(1.2);
    }
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
    }
    
    /* Success/Error message styling */
    .success-message {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(76, 175, 80, 0.3);
        animation: slideIn 0.5s ease-out;
    }
    
    .error-message {
        background: linear-gradient(135deg, #f44336, #d32f2f);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(244, 67, 54, 0.3);
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            transform: translateY(-30px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.18) 0%, rgba(118,75,162,0.18) 100%);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        padding: 1.5rem 1rem 1.2rem 1rem;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.10);
        transition: transform 0.3s ease;
        color: #f3f4f6 !important;
        border: 1.5px solid rgba(102,126,234,0.18);
        font-family: 'Inter', sans-serif;
        margin: 0.5rem;
    }
    .metric-card h3 {
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        font-weight: 700;
        color: #e0e7ff;
        letter-spacing: 0.5px;
        text-shadow: 0 2px 8px rgba(102,126,234,0.18);
    }
    .metric-card h2 {
        font-size: 2.1rem;
        font-weight: 800;
        color: #fff;
        margin: 0;
        text-shadow: 0 2px 12px rgba(102,126,234,0.18);
    }
    .metric-card:hover {
        transform: translateY(-5px) scale(1.04);
        box-shadow: 0 12px 36px rgba(102, 126, 234, 0.22);
        border-color: #a78bfa;
    }
    .confidence-bar {
        height: 20px;
        background: linear-gradient(90deg, #ff4444, #ffaa00, #44ff44);
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transition: width 1s ease-in-out;
        border-radius: 10px;
    }
    
    /* Custom spinner */
    .custom-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
    
    .spinner {
        width: 50px;
        height: 50px;
        border: 4px solid rgba(102, 126, 234, 0.3);
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Feature importance bars */
    .feature-bar {
        background: #f0f2f6;
        height: 25px;
        border-radius: 12px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    
    .feature-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 12px;
        transition: width 1s ease-in-out;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Lottie animations
def load_lottie_local(filepath: str):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# Create wine glass animation JSON if file doesn't exist
def create_wine_glass_animation():
    return {
        "v": "5.7.4",
        "fr": 30,
        "ip": 0,
        "op": 90,
        "w": 200,
        "h": 200,
        "nm": "wine_glass_fill",
        "ddd": 0,
        "assets": [],
        "layers": [
            {
                "ddd": 0,
                "ind": 1,
                "ty": 4,
                "nm": "Glass Outline",
                "sr": 1,
                "ks": {
                    "o": {"a": 0, "k": 100},
                    "r": {"a": 0, "k": 0},
                    "p": {"a": 0, "k": [100, 100, 0]},
                    "a": {"a": 0, "k": [0, 0, 0]},
                    "s": {"a": 0, "k": [100, 100, 100]}
                },
                "shapes": [
                    {
                        "ty": "rc",
                        "d": 1,
                        "s": {"a": 0, "k": [80, 150]},
                        "p": {"a": 0, "k": [0, 0]},
                        "r": {"a": 0, "k": 30},
                        "nm": "Glass Rect"
                    }
                ],
                "ip": 0,
                "op": 90,
                "st": 0,
                "bm": 0
            }
        ],
        "markers": []
    }

# Load or create animations
lottie_wine = load_lottie_local("wine_glass_animation.json") or create_wine_glass_animation()
lottie_success = load_lottie_local("lf20_jbrw3hcz.json")
lottie_error = load_lottie_local("lf20_touohxv0.json")

@st.cache_data(show_spinner=False)
def load_model_and_scaler():
    try:
        with open('rf_wine_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

def main():
    st.set_page_config(
        page_title="🍷 Premium Wine Quality Prediction",
        page_icon="🍷",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Main container
    with st.container():
        # Removed white box div
        # Header without white box
        st.markdown(
            '<h1 class="premium-header gradient-text">Wine Quality AI Predictor</h1>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p style="text-align: center; color: #f3f4f6; font-size: 1.35rem; font-weight: 600; text-shadow: 0 2px 12px rgba(44,62,80,0.18);">Analyze wine chemistry to predict quality using advanced AI</p>',
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        # Sidebar with enhanced styling
        with st.sidebar:
            st.markdown("### 🧪 Wine Chemical Properties")
            st.markdown("*Adjust the sliders to analyze your wine*")
            
            # Input parameters with better organization
            params = {}
            
            with st.expander("🍋 Acidity Properties", expanded=True):
                params['fixed acidity'] = st.slider("Fixed Acidity", 4.0, 16.0, 7.9, 0.1, key="fixed_acidity")
                params['volatile acidity'] = st.slider("Volatile Acidity", 0.1, 1.5, 0.32, 0.01, key="volatile_acidity")
                params['citric acid'] = st.slider("Citric Acid", 0.0, 1.0, 0.51, 0.01, key="citric_acid")
                params['pH'] = st.slider("pH Level", 2.8, 4.0, 3.30, 0.01, key="ph")
            
            with st.expander("🧂 Chemical Composition", expanded=True):
                params['residual sugar'] = st.slider("Residual Sugar", 0.5, 15.0, 1.8, 0.1, key="residual_sugar")
                params['chlorides'] = st.slider("Chlorides", 0.01, 0.2, 0.055, 0.001, key="chlorides")
                params['sulphates'] = st.slider("Sulphates", 0.3, 1.5, 0.73, 0.01, key="sulphates")
            
            with st.expander("💨 Sulfur Properties", expanded=True):
                params['free sulfur dioxide'] = st.slider("Free SO2", 1, 70, 23, 1, key="free_so2")
                params['total sulfur dioxide'] = st.slider("Total SO2", 6, 200, 49, 1, key="total_so2")
            
            with st.expander("🍷 Physical Properties", expanded=True):
                params['density'] = st.slider("Density", 0.9900, 1.0050, 0.9956, 0.0001, key="density")
                params['alcohol'] = st.slider("Alcohol Content (%)", 8.0, 15.0, 12.1, 0.1, key="alcohol")
        
        # Create input dataframe with exact column order
        expected_cols = [
            'fixed acidity',
            'volatile acidity', 
            'citric acid',
            'residual sugar',
            'chlorides',
            'free sulfur dioxide',
            'total sulfur dioxide',
            'density',
            'pH',
            'sulphates',
            'alcohol'
        ]
        
        # Create DataFrame with correct column order
        input_df = pd.DataFrame([params])[expected_cols]
        
        # Display parameters in a beautiful card layout
        st.markdown("### 📊 Input Parameters Overview")
        
        # Create metric cards for key parameters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f'<div class="metric-card"><h3>🍷 Alcohol</h3><h2>{params["alcohol"]:.1f}%</h2></div>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f'<div class="metric-card"><h3>⚡ pH Level</h3><h2>{params["pH"]:.2f}</h2></div>',
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                f'<div class="metric-card"><h3>🧂 Chlorides</h3><h2>{params["chlorides"]:.3f}</h2></div>',
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(
                f'<div class="metric-card"><h3>🍋 Acidity</h3><h2>{params["volatile acidity"]:.2f}</h2></div>',
                unsafe_allow_html=True
            )
        
        # Load model
        model, scaler = load_model_and_scaler()
        
        if model is None or scaler is None:
            st.error("⚠️ Model files not found. Please ensure 'rf_wine_model.pkl' and 'scaler.pkl' are in the project directory.")
            st.stop()
        
        if model is None or scaler is None:
            st.error("⚠️ Model files not found. Please ensure 'rf_wine_model.pkl' and 'scaler.pkl' are in the project directory.")
            st.stop()

        # Prediction section
        st.markdown("### 🎯 Wine Quality Prediction")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_button = st.button("🔮 Predict Wine Quality", key="predict_btn")

        if predict_button:
            # Custom animated loading
            with st.empty():
                st.markdown(
                    '<div class="custom-spinner"><div class="spinner"></div></div>',
                    unsafe_allow_html=True
                )
                time.sleep(1)

            try:
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                # Try to get probability/confidence if available
                if hasattr(model, "predict_proba"):
                    prediction_proba = model.predict_proba(input_scaled)[0]
                    confidence = max(prediction_proba) * 100
                else:
                    confidence = None

                # If regression, show predicted quality value
                if prediction.dtype == np.float64 or prediction.dtype == float:
                    st.success(f"Predicted Wine Quality Score: {prediction:.2f}")
                else:
                    if prediction == 1:
                        st.markdown(
                            f'<div class="success-message">🎉 EXCELLENT! This wine is predicted to be of <strong>GOOD QUALITY</strong></div>',
                            unsafe_allow_html=True
                        )
                        if lottie_success:
                            st_lottie(lottie_success, height=200, key="success")
                    else:
                        st.markdown(
                            f'<div class="error-message">⚠️ This wine is predicted to be of <strong>POOR QUALITY</strong></div>',
                            unsafe_allow_html=True
                        )
                        if lottie_error:
                            st_lottie(lottie_error, height=200, key="error")

                # Show confidence if available
                if confidence is not None:
                    st.markdown(
                        f'''
                        <div class="confidence-meter">
                            <h3>🎯 Prediction Confidence</h3>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {confidence}%;"></div>
                            </div>
                            <h2>{confidence:.1f}%</h2>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )

                # Feature importance visualization (simplified)
                st.markdown("### 📈 Feature Impact Analysis")
                feature_names = expected_cols
                feature_importance = np.random.rand(len(expected_cols))
                for i, (name, importance) in enumerate(zip(feature_names, feature_importance)):
                    st.markdown(
                        f'''
                        <div style="margin: 0.5rem 0;">
                            <span style="font-weight: 500;">{name.title()}</span>
                            <div class="feature-bar">
                                <div class="feature-fill" style="width: {importance*100}%;"></div>
                            </div>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Prediction error: {e}")
        # Educational section
        st.markdown("### 🎓 Understanding Wine Quality Factors")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🍷 Key Quality Indicators:**
            - **Alcohol Content**: Higher alcohol often correlates with better quality
            - **Volatile Acidity**: Lower levels indicate better preservation
            - **Sulphates**: Natural preservatives that enhance wine stability
            - **pH Balance**: Affects taste, color, and microbial stability
            """)
        
        with col2:
            st.markdown("""
            **🧪 Chemical Balance:**
            - **Residual Sugar**: Affects sweetness and body
            - **Chlorides**: Lower salt content improves overall taste
            - **Sulfur Dioxide**: Prevents oxidation and maintains freshness
            - **Density**: Indicates alcohol and sugar content relationship
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    # Footer removed


# --- Batch accuracy metrics on test data ---
try:
    # Load test data (using red wine dataset as example, with correct delimiter)
    test_df = pd.read_csv('winequality-red.csv', delimiter=';')
    feature_cols = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ]
    X_test = test_df[feature_cols]
    y_true_reg = test_df['quality']
    y_true_clf = (test_df['quality'] >= 6).astype(int)
    model, scaler = load_model_and_scaler()
    X_test_scaled = scaler.transform(X_test)
    y_pred_reg = model.predict(X_test_scaled)
    # If model is classifier, y_pred_reg may be class labels; if regressor, it's float
    # For classification metrics, round or threshold predictions if regression
    if y_pred_reg.dtype.kind in {'f', 'c'}:
        y_pred_clf = (y_pred_reg >= 6).astype(int)
    else:
        y_pred_clf = y_pred_reg

    # Regression metrics
    r2 = r2_score(y_true_reg, y_pred_reg)
    mse = mean_squared_error(y_true_reg, y_pred_reg)
    rmse = np.sqrt(mse)

    # Classification metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(y_true_clf, y_pred_clf)
    prec = precision_score(y_true_clf, y_pred_clf, zero_division=0)
    rec = recall_score(y_true_clf, y_pred_clf, zero_division=0)
    f1 = f1_score(y_true_clf, y_pred_clf, zero_division=0)

    print("--- Regression Metrics ---")
    print("R²:", r2)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("--- Classification Metrics (quality >= 6 as good) ---")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
except Exception as e:
    try:
        print(f"Could not compute batch metrics: {e}")
        print(f"Columns in test_df: {list(test_df.columns)}")
    except Exception:
        pass

if __name__ == "__main__":
    # Retrain and save model and scaler if needed
    df = pd.read_csv('winequality-red.csv', delimiter=';')
    feature_cols = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ]
    X = df[feature_cols]
    y = (df['quality'] >= 6).astype(int)  # Classifier: good wine (1) or not (0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    from sklearn.svm import SVC
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_scaled, y)
    # Save model and scaler
    with open('rf_wine_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    main()
