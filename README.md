# 🍷 Wine Quality AI Predictor

An interactive Streamlit app that predicts the quality of red wine based on 11 chemical properties using a Support Vector Machine (SVM) model. The app features a premium UI, instant Good/Poor prediction, and confidence metrics.

---

## 1) Overview

- **Technology:** Python, Streamlit, scikit-learn
- **Artifacts:**  
  - `rf_wine_model.pkl` — SVM Classifier  
  - `scaler.pkl` — StandardScaler
- **Dataset:** UCI Red Wine Quality (`winequality-red.csv`, semicolon `;` delimited)
- **Purpose:** Assistive decision support for wine quality assessment

---

## 2) Features

- Modern UI with custom CSS, metric cards, and Lottie animations
- Sidebar sliders for 11 chemical inputs
- Automated scaling & prediction using SVM (SVC)
- Good/Poor result with optional confidence bar
- Auto-retrain on first run if model/scaler are missing

---

## 3) Project Structure

```
wine_quality_app.py         # Streamlit application (UI + prediction + metrics)
winequality-red.csv         # Dataset (semicolon `;` delimited)
rf_wine_model.pkl           # Trained SVM classifier (auto-generated if missing)
scaler.pkl                  # Fitted StandardScaler (auto-generated if missing)
lf20_jbrw3hcz.json          # Lottie animation
lf20_touohxv0.json          # Lottie animation
wine_glass_fill.json        # Local animation JSON
```

---

## 4) Dataset & Features

**Features (11 numerical):**
- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol

**Target:**  
- Classification: _Good_ (quality ≥ 6), _Poor_ (< 6)

---

## 5) How It Works

1. User sets values via sidebar sliders → forms a `pandas.DataFrame` (fixed column order).
2. Inputs are standardized using `scaler.pkl`.
3. Model predicts class and (if supported) class probabilities.
4. UI displays Good/Poor status, confidence bar, and simple feature impact info.

---

## 6) Setup & Run (Windows PowerShell Example)

**Activate your virtual environment:**
```powershell
& "E:/Wine ML/.venv/Scripts/Activate.ps1"
```

**Install dependencies:**
```powershell
pip install streamlit scikit-learn pandas numpy streamlit-lottie
```

**Run the Streamlit app:**
```powershell
streamlit run "e:\\Wine ML\\wine_quality_app.py"
```

**Run as a plain Python script (for dev/testing):**
```powershell
& "E:/Wine ML/.venv/Scripts/python.exe" "e:\\Wine ML\\wine_quality_app.py"
```

*Note: Prefer `streamlit run` for full UI experience.*

---

## 7) Example Input Values (Good Red Wine)

| Feature               | Value Range or Example      | Notes               |
|-----------------------|----------------------------|---------------------|
| fixed acidity         | 7.4                        |                     |
| volatile acidity      | 0.26 (lower is better)     |                     |
| citric acid           | 0.34                       |                     |
| residual sugar        | 2.0                        |                     |
| chlorides             | 0.045–0.060                |                     |
| free sulfur dioxide   | 15–25                      |                     |
| total sulfur dioxide  | 40–60                      |                     |
| density               | 0.994–0.996                |                     |
| pH                    | 3.2–3.4                    |                     |
| sulphates             | 0.6–0.8                    |                     |
| alcohol               | 11.5–12.8                  |                     |

---

## 8) Model & Metrics

- **Model:** `SVC(kernel='rbf', probability=True, random_state=42)`
- **Scaling:** StandardScaler over all 11 features
- **Tuning:** Sample cross-validation runs for regressor/classifier variants

**Sample Metrics (illustrative):**
- Random Forest (CV): Accuracy ~0.78, F1 ~0.80
- Regressor: R² ~0.53, RMSE ~0.55
- Naive full-dataset evaluation: Accuracy ~0.60, F1 ~0.40

_Recommendation: Use proper train/test or cross-validation. Consider probability calibration (Platt/Isotonic)._

---

## 9) Architecture (Text Diagram)

```
User (Streamlit UI sliders)
       │
       ▼
Pandas DataFrame (11 features, fixed order)
       │
       ▼
StandardScaler (scaler.pkl)
       │
       ▼
SVM Classifier (rf_wine_model.pkl)
       │
       ▼
Good/Poor + Confidence → UI cards, message, Lottie animation
```

---

## 10) Troubleshooting

- **Model files not found:** First run will retrain and save `rf_wine_model.pkl` and `scaler.pkl`.
- **CSV delimiter issues:** Use `delimiter=';'` for `winequality-red.csv`.
- **Predict button shows nothing:** Ensure model/scaler load OK and feature order matches `expected_cols` in code.
- **Streamlit warnings:** Expected when running as plain Python; use `streamlit run` for best UI.

---

## 11) PPT Outline

1. Title & Team — Wine Quality AI Predictor (Python/Streamlit/SVM)
2. Problem Overview — subjective assessments; fast, consistent aid
3. Dataset & Features — 11 chemical properties; UCI red wine
4. Target Definition — Good (≥6) vs Poor; why binary
5. Solution Architecture — UI → Scale → SVM Classifier → Result
6. Modeling Approach — preprocessing, SVM params, tuning overview
7. Performance — CV highlights vs naive full-dataset caveat
8. App UI & Flow — sliders, metric cards, predict button, confidence
9. Live Demo Plan — set values (alcohol ~12%, VA low, sulphates 0.6–0.8); click predict
10. Limitations & Risks — dataset, calibration, SHAP, ethics
11. Roadmap — proper evaluation, calibration, multi-class, deployment
12. Takeaways — interactive tool; baseline ready to improve

---

## 12) Slide Content Draft (Bullets & Speaker Notes)

- Each slide: 3–5 bullets + a one-sentence talk track
- See companion notes in your chat for detailed content that can be pasted directly into PPT speaker notes.

---

## 13) Screenshot Checklist

- Sidebar with expanders (Acidity, Chemical Composition, Sulfur, Physical)
- Metric cards row: Alcohol %, pH, Chlorides, Volatile acidity
- Prediction state (Good) with green card + Lottie
- Prediction state (Poor) with red card + Lottie
- Confidence meter bar

---

## 14) Limitations & Roadmap

**Limitations**
- Small dataset; class thresholding may not align with all domains
- Uncalibrated probabilities; UI feature impact is illustrative

**Roadmap**
- Cross-validated evaluation & calibration
- Feature attribution (SHAP)
- Hyperparameter tuning & model registry
- Support for white wine / multi-class quality
- Packaging & deployment (Docker/Cloud)

---


