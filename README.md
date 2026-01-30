# HealthPredict AI - Multi-Disease Prediction System

A production-ready, enterprise-grade web application for health risk assessment powered by machine learning. Built with Streamlit and designed for deployment on Streamlit Cloud.

## Features

### Disease Prediction Modules

1. **Diabetes Risk Assessment** (PIMA Indians Dataset)
   - 8 health parameters analyzed
   - Glucose, BMI, blood pressure, insulin levels, and more

2. **Heart Disease Check** (UCI Heart Disease Dataset)
   - 13 cardiac parameters analyzed
   - Cholesterol, ECG results, exercise tests, and more

3. **Parkinson's Disease Screening** (Voice Measurements Dataset)
   - 22 voice biomarkers analyzed
   - Jitter, shimmer, harmonics, and nonlinear measures

4. **Liver Health Analysis** (Indian Liver Patient Dataset)
   - 10 liver function parameters analyzed
   - Bilirubin, enzymes, proteins, and ratios

### Technical Features

- **Machine Learning Pipeline**: Automated model training, comparison, and selection
- **Multiple Algorithms**: Logistic Regression, Random Forest, Gradient Boosting, SVM
- **Smart Caching**: Models trained once and cached for performance
- **Cross-Validation**: Robust model evaluation with 5-fold CV
- **Real-time Predictions**: Instant risk assessment with confidence scores
- **Professional UI**: Healthcare-themed design with responsive layout

## Project Structure

```
multiple-disease-prediction-web-app/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── data/
│   ├── diabetes.csv           # PIMA Indians Diabetes Dataset
│   ├── Heart_Disease_Prediction.csv  # UCI Heart Disease Dataset
│   ├── parkinsons.csv         # Parkinson's Voice Dataset
│   └── indian_liver_patient.csv      # Indian Liver Patient Dataset
├── models/                    # Saved trained models (auto-generated)
└── src/
    ├── __init__.py
    ├── styles.py              # Custom CSS and UI components
    ├── ml_utils.py            # ML pipeline utilities
    ├── home.py                # Home page component
    ├── diabetes_predictor.py  # Diabetes prediction module
    ├── heart_predictor.py     # Heart disease prediction module
    ├── parkinsons_predictor.py # Parkinson's prediction module
    └── liver_predictor.py     # Liver disease prediction module
```

## Local Development

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd multiple-disease-prediction-web-app
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

5. Open your browser and navigate to `http://localhost:8501`

## Deployment on Streamlit Cloud

### Step 1: Prepare Repository

1. Push your code to a GitHub repository
2. Ensure all files are committed:
   - `app.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
   - `src/` directory with all modules
   - `data/` directory with all CSV files

### Step 2: Deploy on Streamlit Cloud

1. Go to [Streamlit Cloud](https://share.streamlit.io/)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set the main file path: `app.py`
6. Click "Deploy"

### Step 3: Configuration (Optional)

- The app uses `.streamlit/config.toml` for theme settings
- No additional secrets or environment variables required
- Models are trained automatically on first run

## Machine Learning Models

### Model Training Process

1. **Data Loading**: CSV files loaded with proper preprocessing
2. **Feature Scaling**: StandardScaler normalization
3. **Model Training**: Multiple algorithms trained and compared
4. **Model Selection**: Best model selected based on cross-validation score
5. **Caching**: Models cached using `st.cache_resource`

### Algorithms Used

| Disease | Primary Model | Alternative Models |
|---------|--------------|-------------------|
| Diabetes | Random Forest | Logistic Regression, Gradient Boosting |
| Heart Disease | Random Forest | Logistic Regression, Gradient Boosting |
| Parkinson's | SVM | Logistic Regression, Random Forest |
| Liver Disease | Random Forest | Logistic Regression, Gradient Boosting |

### Model Performance

Models are evaluated using:
- Train/Test split (80/20)
- 5-fold Cross-Validation
- Accuracy, precision, and recall metrics

## API Reference

### ML Utilities (`src/ml_utils.py`)

```python
# Load and preprocess data
load_diabetes_data() -> pd.DataFrame
load_heart_data() -> pd.DataFrame
load_parkinsons_data() -> pd.DataFrame
load_liver_data() -> pd.DataFrame

# Train models
train_diabetes_model() -> tuple
train_heart_model() -> tuple
train_parkinsons_model() -> tuple
train_liver_model() -> tuple

# Make predictions
make_prediction(model, scaler, features) -> tuple
get_risk_level(probability, prediction) -> str
```

## Customization

### Modifying the Theme

Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1e3a5f"
backgroundColor = "#f8fafc"
secondaryBackgroundColor = "#ffffff"
textColor = "#1e293b"
```

### Adding New Disease Models

1. Add dataset to `data/` directory
2. Create preprocessing function in `ml_utils.py`
3. Create new predictor module in `src/`
4. Add navigation entry in `app.py`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for educational and informational purposes only.

## Disclaimer

**Important Medical Disclaimer:**

This application provides informational insights only and is not a medical diagnosis. The predictions are based on machine learning models trained on historical data and should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

## Support

For issues and feature requests, please open an issue on GitHub.

---

Built with Streamlit | Powered by scikit-learn | Designed for Healthcare
