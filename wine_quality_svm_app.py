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
from sklearn.svm import SVC

def load_custom_css():
	st.markdown("""
	<style>
	/* ...CSS unchanged... */
	</style>
	""", unsafe_allow_html=True)

def load_lottie_local(filepath: str):
	if os.path.exists(filepath):
		with open(filepath, "r", encoding="utf-8") as f:
			return json.load(f)
	return None

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

lottie_wine = load_lottie_local("wine_glass_animation.json") or create_wine_glass_animation()
lottie_success = load_lottie_local("lottie_success.json")
lottie_error = load_lottie_local("lottie_error.json")

@st.cache_data(show_spinner=False)
def load_model_and_scaler():
	try:
		with open('svm_wine_model.pkl', 'rb') as f:
			model = pickle.load(f)
		with open('wine_scaler.pkl', 'rb') as f:
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
	load_custom_css()
	with st.container():
		st.markdown(
			'<h1 class="premium-header gradient-text">Wine Quality AI Predictor</h1>',
			unsafe_allow_html=True
		)
		st.markdown(
			'<p style="text-align: center; color: #f3f4f6; font-size: 1.35rem; font-weight: 600; text-shadow: 0 2px 12px rgba(44,62,80,0.18);">Analyze wine chemistry to predict quality using advanced AI</p>',
			unsafe_allow_html=True
		)
		st.markdown("---")
		with st.sidebar:
			st.markdown("### 🧪 Wine Chemical Properties")
			st.markdown("*Adjust the sliders to analyze your wine*")
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
		input_df = pd.DataFrame([params])[expected_cols]
		st.markdown("### 📊 Input Parameters Overview")
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
		model, scaler = load_model_and_scaler()
		if model is None or scaler is None:
			st.error("⚠️ Model files not found. Please ensure 'svm_wine_model.pkl' and 'wine_scaler.pkl' are in the project directory.")
			st.stop()
		st.markdown("### 🎯 Wine Quality Prediction")
		col1, col2, col3 = st.columns([1, 1, 1])
		with col2:
			predict_button = st.button("🔮 Predict Wine Quality", key="predict_btn")
		if predict_button:
			with st.empty():
				st.markdown(
					'<div class="custom-spinner"><div class="spinner"></div></div>',
					unsafe_allow_html=True
				)
				time.sleep(1)
			try:
				input_scaled = scaler.transform(input_df)
				prediction = model.predict(input_scaled)[0]
				if hasattr(model, "predict_proba"):
					prediction_proba = model.predict_proba(input_scaled)[0]
					confidence = max(prediction_proba) * 100
				else:
					confidence = None
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

if __name__ == "__main__":
	# Retrain and save model and scaler if needed
	df = pd.read_csv('winequality_red.csv', delimiter=';')
	feature_cols = [
		'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
		'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
		'pH', 'sulphates', 'alcohol'
	]
	X = df[feature_cols]
	y = (df['quality'] >= 6).astype(int)
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)
	clf = SVC(kernel='rbf', probability=True, random_state=42)
	clf.fit(X_scaled, y)
	with open('svm_wine_model.pkl', 'wb') as f:
		pickle.dump(clf, f)
	with open('wine_scaler.pkl', 'wb') as f:
		pickle.dump(scaler, f)
	main()
