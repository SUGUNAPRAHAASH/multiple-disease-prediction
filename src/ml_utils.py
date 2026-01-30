"""
Machine Learning Utilities for Multi-Disease Prediction System.

This module contains all ML-related functions including:
- Data preprocessing
- Model training
- Model evaluation
- Prediction utilities

All functions are optimized for Streamlit with proper caching.
"""

import sys
from pathlib import Path
# Add project root to path for Streamlit Cloud compatibility
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import streamlit as st
from pathlib import Path


# ============================================
# PATH UTILITIES
# ============================================

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_data_path(filename: str) -> Path:
    """Get the path to a data file."""
    # Try multiple locations for flexibility
    possible_paths = [
        get_project_root() / "data" / filename,
        get_project_root() / filename,
        Path("data") / filename,
        Path(filename)
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # Return the default path (data folder)
    return get_project_root() / "data" / filename


def get_model_path(filename: str) -> Path:
    """Get the path to save/load a model."""
    models_dir = get_project_root() / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir / filename


# ============================================
# DATA LOADING & PREPROCESSING
# ============================================

@st.cache_data(ttl=3600)
def load_diabetes_data() -> pd.DataFrame:
    """
    Load and preprocess the PIMA Indians Diabetes dataset.

    Returns:
        pd.DataFrame: Preprocessed diabetes dataset
    """
    df = pd.read_csv(get_data_path("diabetes.csv"))

    # Replace zero values with NaN for columns where 0 is not valid
    # then impute with median
    zero_not_valid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_not_valid:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    return df


@st.cache_data(ttl=3600)
def load_heart_data() -> pd.DataFrame:
    """
    Load and preprocess the UCI Heart Disease dataset.

    Returns:
        pd.DataFrame: Preprocessed heart disease dataset
    """
    df = pd.read_csv(get_data_path("Heart_Disease_Prediction.csv"))

    # Encode target variable
    df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})

    # Handle missing values if any
    df = df.fillna(df.median(numeric_only=True))

    return df


@st.cache_data(ttl=3600)
def load_parkinsons_data() -> pd.DataFrame:
    """
    Load and preprocess the Parkinson's Disease dataset.

    Returns:
        pd.DataFrame: Preprocessed Parkinson's dataset
    """
    df = pd.read_csv(get_data_path("parkinsons.csv"))

    # Drop the 'name' column as it's just an identifier
    if 'name' in df.columns:
        df = df.drop('name', axis=1)

    return df


@st.cache_data(ttl=3600)
def load_liver_data() -> pd.DataFrame:
    """
    Load and preprocess the Indian Liver Patient dataset.

    Returns:
        pd.DataFrame: Preprocessed liver disease dataset
    """
    df = pd.read_csv(get_data_path("indian_liver_patient.csv"))

    # Encode Gender
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))

    # Convert target: 1 = Liver Disease, 2 = No Liver Disease
    # We'll convert to 1 = Disease, 0 = No Disease
    df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})

    return df


# ============================================
# MODEL TRAINING UTILITIES
# ============================================

def train_and_evaluate_models(X: np.ndarray, y: np.ndarray,
                             model_configs: dict = None) -> dict:
    """
    Train multiple models and return their performance metrics.

    Args:
        X: Feature matrix
        y: Target vector
        model_configs: Dictionary of model configurations

    Returns:
        dict: Dictionary containing trained models and their metrics
    """
    if model_configs is None:
        model_configs = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    for name, model in model_configs.items():
        # Train model
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_test': y_test,
            'y_pred': y_pred
        }

    return results, scaler


def select_best_model(results: dict) -> tuple:
    """
    Select the best performing model based on cross-validation score.

    Args:
        results: Dictionary of model results

    Returns:
        tuple: (best_model_name, best_model, best_accuracy)
    """
    best_name = max(results, key=lambda x: results[x]['cv_mean'])
    best_model = results[best_name]['model']
    best_accuracy = results[best_name]['accuracy']

    return best_name, best_model, best_accuracy


# ============================================
# DISEASE-SPECIFIC MODEL TRAINING
# ============================================

@st.cache_resource(ttl=3600)
def train_diabetes_model():
    """
    Train and cache the diabetes prediction model.

    Returns:
        tuple: (best_model, scaler, model_name, accuracy, all_results)
    """
    df = load_diabetes_data()

    # Features and target
    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values
    feature_names = df.drop('Outcome', axis=1).columns.tolist()

    # Train models
    results, scaler = train_and_evaluate_models(X, y)

    # Select best model
    best_name, best_model, best_accuracy = select_best_model(results)

    # Save model and scaler
    joblib.dump(best_model, get_model_path('diabetes_model.pkl'))
    joblib.dump(scaler, get_model_path('diabetes_scaler.pkl'))

    return best_model, scaler, best_name, best_accuracy, results, feature_names


@st.cache_resource(ttl=3600)
def train_heart_model():
    """
    Train and cache the heart disease prediction model.

    Returns:
        tuple: (best_model, scaler, model_name, accuracy, all_results)
    """
    df = load_heart_data()

    # Features and target
    X = df.drop('Heart Disease', axis=1).values
    y = df['Heart Disease'].values
    feature_names = df.drop('Heart Disease', axis=1).columns.tolist()

    # Train models
    results, scaler = train_and_evaluate_models(X, y)

    # Select best model
    best_name, best_model, best_accuracy = select_best_model(results)

    # Save model and scaler
    joblib.dump(best_model, get_model_path('heart_model.pkl'))
    joblib.dump(scaler, get_model_path('heart_scaler.pkl'))

    return best_model, scaler, best_name, best_accuracy, results, feature_names


@st.cache_resource(ttl=3600)
def train_parkinsons_model():
    """
    Train and cache the Parkinson's disease prediction model.

    Returns:
        tuple: (best_model, scaler, model_name, accuracy, all_results)
    """
    df = load_parkinsons_data()

    # Features and target (status is the target column)
    X = df.drop('status', axis=1).values
    y = df['status'].values
    feature_names = df.drop('status', axis=1).columns.tolist()

    # Train models with SVM added (works well for Parkinson's)
    model_configs = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }

    results, scaler = train_and_evaluate_models(X, y, model_configs)

    # Select best model
    best_name, best_model, best_accuracy = select_best_model(results)

    # Save model and scaler
    joblib.dump(best_model, get_model_path('parkinsons_model.pkl'))
    joblib.dump(scaler, get_model_path('parkinsons_scaler.pkl'))

    return best_model, scaler, best_name, best_accuracy, results, feature_names


@st.cache_resource(ttl=3600)
def train_liver_model():
    """
    Train and cache the liver disease prediction model.

    Returns:
        tuple: (best_model, scaler, model_name, accuracy, all_results)
    """
    df = load_liver_data()

    # Features and target
    X = df.drop('Dataset', axis=1).values
    y = df['Dataset'].values
    feature_names = df.drop('Dataset', axis=1).columns.tolist()

    # Train models
    results, scaler = train_and_evaluate_models(X, y)

    # Select best model
    best_name, best_model, best_accuracy = select_best_model(results)

    # Save model and scaler
    joblib.dump(best_model, get_model_path('liver_model.pkl'))
    joblib.dump(scaler, get_model_path('liver_scaler.pkl'))

    return best_model, scaler, best_name, best_accuracy, results, feature_names


# ============================================
# PREDICTION UTILITIES
# ============================================

def make_prediction(model, scaler, features: np.ndarray) -> tuple:
    """
    Make a prediction using the trained model.

    Args:
        model: Trained ML model
        scaler: Fitted StandardScaler
        features: Input features as numpy array

    Returns:
        tuple: (prediction, probability)
    """
    # Reshape if necessary
    if features.ndim == 1:
        features = features.reshape(1, -1)

    # Scale features
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features_scaled)[0]

    # Get probability if available
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(features_scaled)[0]
        # Return probability of the predicted class
        prob = probability[1] if prediction == 1 else probability[0]
    else:
        prob = None

    return int(prediction), prob


def get_risk_level(probability: float, prediction: int) -> str:
    """
    Determine risk level based on prediction probability.

    Args:
        probability: Prediction probability
        prediction: Binary prediction (0 or 1)

    Returns:
        str: Risk level ('Low', 'Medium', or 'High')
    """
    if prediction == 0:
        if probability >= 0.8:
            return "Low"
        elif probability >= 0.6:
            return "Low"
        else:
            return "Medium"
    else:  # prediction == 1
        if probability >= 0.8:
            return "High"
        elif probability >= 0.6:
            return "Medium"
        else:
            return "Medium"


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Get feature importance from the model.

    Args:
        model: Trained ML model
        feature_names: List of feature names

    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return None

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    return importance_df
