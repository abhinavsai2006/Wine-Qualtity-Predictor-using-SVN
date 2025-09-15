
# 🍷 Wine Quality AI Predictor

An interactive Streamlit app that predicts red wine quality from 11 chemical properties using a Support Vector Machine (SVM) model. The app offers a premium UI, instant Good/Poor prediction with a confidence meter, and learning content about each factor.

## 1) Overview
- Tech: Python, Streamlit, scikit-learn
- Models/Artifacts: `rf_wine_model.pkl` (SVC - Support Vector Machine Classifier), `scaler.pkl` (StandardScaler)
- Dataset: UCI Red Wine Quality (`winequality-red.csv`, semicolon `;` delimited)
- Purpose: Assistive decision support for wine quality assessment

## 2) Features
- Modern UI with custom CSS, metric cards, and animations (Lottie)
- Sidebar sliders for 11 chemical inputs
- Scaling via `StandardScaler` and prediction via Support Vector Machine (SVC)
- Good/Poor result with an optional confidence bar
- Auto-retrain on first run if model/scaler are missing

## 3) Project Structure
- `wine_quality_app.py` — Streamlit application (UI + prediction + metrics printouts)
- `winequality-red.csv` — dataset (semicolon `;` delimited)
- `rf_wine_model.pkl` — trained SVM classifier (generated if missing)
- `scaler.pkl` — fitted `StandardScaler` (generated if missing)
- `lf20_jbrw3hcz.json`, `lf20_touohxv0.json` — Lottie animations
- `wine_glass_fill.json` — local animation JSON

## 4) Dataset & Features
The app uses 11 numerical features:
- `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`
- `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`

Target used by the app: Classification (Good = quality ≥ 6, Poor = < 6).

## 5) How it works
1. User sets values in the sidebar → a `pandas.DataFrame` is created with a fixed column order.
2. Inputs are standardized using `scaler.pkl`.
3. `rf_wine_model.pkl` predicts class and (if supported) class probabilities.
4. UI shows Good/Poor message, confidence bar, and a simple feature-impact section.

## 6) Setup & Run (Windows PowerShell)
Activate your virtual environment (if it exists):

```powershell
& "E:/Wine ML/.venv/Scripts/Activate.ps1"
```

Install packages (if needed):

```powershell
pip install streamlit scikit-learn pandas numpy streamlit-lottie
```

Run as a Streamlit app (recommended):

```powershell
streamlit run "e:\\Wine ML\\wine_quality_app.py"
```

Run as a plain Python script (dev/testing):

```powershell
& "E:/Wine ML/.venv/Scripts/python.exe" "e:\\Wine ML\\wine_quality_app.py"
```

Note: Running as a plain Python script prints metrics in the terminal and may show Streamlit script context warnings; use `streamlit run` for full UI.

## 7) Example input values (typically good red wine)
- `fixed acidity`: 7.4
- `volatile acidity`: 0.26 (lower is better)
- `citric acid`: 0.34
- `residual sugar`: 2.0
- `chlorides`: 0.045–0.060
- `free sulfur dioxide`: 15–25
- `total sulfur dioxide`: 40–60
- `density`: 0.994–0.996
- `pH`: 3.2–3.4
- `sulphates`: 0.6–0.8
- `alcohol`: 11.5–12.8

## 8) Model & Metrics (context)
- Model in app: `SVC(kernel='rbf', probability=True, random_state=42)`
- Scaling: `StandardScaler` over all 11 features
- Tuning (in code) shows sample CV runs for regressor/classifier variants.

Observed metrics from dev runs (illustrative):
- During tuning (CV examples printed): SVM Classifier Accuracy and F1 will vary; typical values for Random Forest were Accuracy ~0.78, F1 ~0.80; Regressor R² ~0.53, RMSE ~0.55
- Naive full-dataset evaluation (not a proper test): Accuracy ~0.60, F1 ~0.40

Recommendation: Use proper train/test or cross-validated evaluation for reporting; consider probability calibration (Platt/Isotonic).

## 9) Architecture (text diagram)
```
User (Streamlit UI sliders)
       │
       ▼
Pandas DataFrame (fixed column order: 11 features)
       │
       ▼
StandardScaler (scaler.pkl)
       │
       ▼
SVM Classifier (rf_wine_model.pkl)
       │
       ▼
Good/Poor + Confidence  →  UI cards, message, Lottie animation
```

## 10) Troubleshooting
- "Model files not found" → first run will retrain and save `rf_wine_model.pkl` and `scaler.pkl`.
- CSV delimiter issues → use `delimiter=';'` for `winequality-red.csv`.
- Predict button shows nothing → ensure model/scaler load OK and the feature order matches `expected_cols` in code.
- Streamlit ScriptRunContext warnings → expected in plain Python; use `streamlit run` for UI.

## 11) PPT outline (ready to copy)
1. Title & Team — Wine Quality AI Predictor (Python/Streamlit/RandomForest)
2. Problem Overview — subjective assessments; fast, consistent aid
3. Dataset & Features — 11 chemical properties; UCI red wine
4. Target Definition — Good (≥6) vs Poor; why binary
5. Solution Architecture — UI → Scale → RF Classifier → Result
6. Modeling Approach — preprocessing, RF params, tuning overview
7. Performance — CV highlights vs naive full-dataset caveat
8. App UI & Flow — sliders, metric cards, predict button, confidence
9. Live Demo Plan — set values (alcohol ~12%, VA low, sulphates 0.6–0.8); click predict
10. Limitations & Risks — dataset, calibration, SHAP, ethics
11. Roadmap — proper evaluation, calibration, multi-class, deployment
12. Takeaways — interactive tool; baseline ready to improve

## 12) Slide content draft (bullets + speaker notes)
- Each slide: 3–5 bullets and a one-sentence talk track
- See the companion notes in your chat for detailed copy; you can paste them directly into PPT speaker notes.

## 13) Screenshot checklist
- Sidebar with expanders (Acidity, Chemical Composition, Sulfur, Physical)
- Metric cards row: Alcohol %, pH, Chlorides, Volatile acidity
- Prediction state (Good) with green card + Lottie
- Prediction state (Poor) with red card + Lottie
- Confidence meter bar

## 14) Limitations & Roadmap
**Limitations**
- Small dataset; class thresholding may not align with all domains
- Uncalibrated probabilities; UI feature-impact is illustrative

**Roadmap**
- Cross-validated evaluation + calibration
- True feature attribution (e.g., SHAP)
- Hyperparameter sweeps + model registry
- Support white wine / multi-class quality
- Packaging & deployment (Docker/Cloud)

---
For educational/demo purposes. Verify dataset licensing before redistribution.
