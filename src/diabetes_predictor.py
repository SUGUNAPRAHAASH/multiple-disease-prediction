"""
Diabetes Prediction Module for Multi-Disease Prediction System.

This module handles:
- Diabetes risk prediction using PIMA Indians dataset
- User input form with proper validation
- Result visualization and interpretation
"""

import sys
from pathlib import Path
# Add project root to path for Streamlit Cloud compatibility
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.ml_utils import (
    train_diabetes_model,
    make_prediction,
    get_risk_level,
    get_feature_importance
)
from src.styles import render_header, render_disclaimer, render_result_card


def get_diabetes_explanation(prediction: int, probability: float, inputs: dict) -> str:
    """
    Generate a personalized explanation based on the prediction and input values.

    Args:
        prediction: Binary prediction (0 or 1)
        probability: Prediction probability
        inputs: Dictionary of input values

    Returns:
        str: Explanation text
    """
    if prediction == 1:
        explanation = "Based on your health parameters, the model has identified potential risk factors for diabetes. "

        # Analyze specific risk factors
        risk_factors = []
        if inputs['glucose'] > 140:
            risk_factors.append("elevated glucose levels")
        if inputs['bmi'] > 30:
            risk_factors.append("BMI in the obese range")
        if inputs['blood_pressure'] > 90:
            risk_factors.append("elevated blood pressure")
        if inputs['age'] > 45:
            risk_factors.append("age-related risk factors")
        if inputs['dpf'] > 0.5:
            risk_factors.append("family history indicators")

        if risk_factors:
            explanation += f"Key factors include: {', '.join(risk_factors)}. "

        explanation += "We recommend consulting with a healthcare provider for proper evaluation and discussing lifestyle modifications."
    else:
        explanation = "Your health parameters suggest a lower risk profile for diabetes. "
        explanation += "Continue maintaining healthy lifestyle habits including regular exercise, balanced diet, and routine health check-ups."

    return explanation


def render_model_info(model_name: str, accuracy: float, results: dict):
    """Render model information and performance metrics."""

    st.markdown("""
        <h4 style="color: #1e3a5f; margin-bottom: 1rem;">
            📊 Model Performance
        </h4>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Selected Model", model_name)

    with col2:
        st.metric("Test Accuracy", f"{accuracy * 100:.1f}%")

    with col3:
        cv_score = results[model_name]['cv_mean']
        st.metric("Cross-Val Score", f"{cv_score * 100:.1f}%")

    # Model comparison chart
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] * 100 for m in model_names]
    cv_scores = [results[m]['cv_mean'] * 100 for m in model_names]

    fig = go.Figure(data=[
        go.Bar(name='Test Accuracy', x=model_names, y=accuracies, marker_color='#3b82f6'),
        go.Bar(name='CV Score', x=model_names, y=cv_scores, marker_color='#10b981')
    ])

    fig.update_layout(
        barmode='group',
        title='Model Comparison',
        yaxis_title='Score (%)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1e3a5f'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_diabetes_page():
    """Render the diabetes prediction page."""

    # Header
    render_header(
        title="Diabetes Risk Assessment",
        subtitle="Evaluate your risk of Type 2 Diabetes using the PIMA Indians dataset model",
        icon="🩸"
    )

    # Load/train model with spinner
    with st.spinner("Loading diabetes prediction model..."):
        model, scaler, model_name, accuracy, results, feature_names = train_diabetes_model()

    # Main content layout
    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("""
            <div class="health-card">
                <h3 style="color: #1e3a5f; margin-bottom: 1.5rem;">
                    📋 Enter Your Health Parameters
                </h3>
            </div>
        """, unsafe_allow_html=True)

        with st.form("diabetes_form"):
            # Row 1
            col1, col2 = st.columns(2)
            with col1:
                pregnancies = st.number_input(
                    "Pregnancies",
                    min_value=0,
                    max_value=20,
                    value=1,
                    help="Number of times pregnant"
                )
            with col2:
                glucose = st.number_input(
                    "Glucose (mg/dL)",
                    min_value=0,
                    max_value=300,
                    value=120,
                    help="Plasma glucose concentration (2 hours in an oral glucose tolerance test)"
                )

            # Row 2
            col3, col4 = st.columns(2)
            with col3:
                blood_pressure = st.number_input(
                    "Blood Pressure (mm Hg)",
                    min_value=0,
                    max_value=200,
                    value=70,
                    help="Diastolic blood pressure"
                )
            with col4:
                skin_thickness = st.number_input(
                    "Skin Thickness (mm)",
                    min_value=0,
                    max_value=100,
                    value=20,
                    help="Triceps skin fold thickness"
                )

            # Row 3
            col5, col6 = st.columns(2)
            with col5:
                insulin = st.number_input(
                    "Insulin (μU/mL)",
                    min_value=0,
                    max_value=900,
                    value=80,
                    help="2-Hour serum insulin"
                )
            with col6:
                bmi = st.number_input(
                    "BMI (kg/m²)",
                    min_value=0.0,
                    max_value=70.0,
                    value=25.0,
                    step=0.1,
                    help="Body Mass Index"
                )

            # Row 4
            col7, col8 = st.columns(2)
            with col7:
                dpf = st.number_input(
                    "Diabetes Pedigree Function",
                    min_value=0.0,
                    max_value=3.0,
                    value=0.5,
                    step=0.01,
                    help="Diabetes pedigree function (genetic influence score)"
                )
            with col8:
                age = st.number_input(
                    "Age (years)",
                    min_value=1,
                    max_value=120,
                    value=30,
                    help="Age in years"
                )

            # Submit button
            submitted = st.form_submit_button(
                "🔍 Analyze Risk",
                use_container_width=True,
                type="primary"
            )

    with col_result:
        if submitted:
            # Prepare input features
            features = np.array([
                pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, dpf, age
            ])

            # Make prediction
            prediction, probability = make_prediction(model, scaler, features)

            # Determine risk level
            risk_level = get_risk_level(probability, prediction)

            # Get explanation
            inputs = {
                'glucose': glucose,
                'bmi': bmi,
                'blood_pressure': blood_pressure,
                'age': age,
                'dpf': dpf
            }
            explanation = get_diabetes_explanation(prediction, probability, inputs)

            # Display result
            render_result_card(
                prediction=prediction,
                disease_name="Diabetes",
                probability=probability,
                risk_level=risk_level,
                explanation=explanation
            )

            # Input summary
            st.markdown("---")
            st.markdown("""
                <h4 style="color: #1e3a5f; margin: 1rem 0;">
                    📋 Your Input Summary
                </h4>
            """, unsafe_allow_html=True)

            input_df = pd.DataFrame({
                'Parameter': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
                             'Insulin', 'BMI', 'Diabetes Pedigree', 'Age'],
                'Value': [pregnancies, f"{glucose} mg/dL", f"{blood_pressure} mm Hg",
                         f"{skin_thickness} mm", f"{insulin} μU/mL", f"{bmi:.1f} kg/m²",
                         f"{dpf:.3f}", f"{age} years"]
            })

            st.dataframe(input_df, use_container_width=True, hide_index=True)

        else:
            # Show placeholder when no prediction yet
            st.markdown("""
                <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                            border-radius: 16px; padding: 3rem; text-align: center;
                            border: 2px dashed #93c5fd;">
                    <span style="font-size: 4rem;">🔬</span>
                    <h3 style="color: #1e3a5f; margin: 1rem 0;">Ready to Analyze</h3>
                    <p style="color: #64748b;">
                        Enter your health parameters on the left and click
                        <strong>"Analyze Risk"</strong> to get your diabetes risk assessment.
                    </p>
                </div>
            """, unsafe_allow_html=True)

    # Model Information Section
    st.markdown("---")

    with st.expander("📊 View Model Information & Performance"):
        render_model_info(model_name, accuracy, results)

        # Feature importance
        importance_df = get_feature_importance(model, feature_names)
        if importance_df is not None:
            st.markdown("""
                <h4 style="color: #1e3a5f; margin: 1.5rem 0 1rem 0;">
                    🎯 Feature Importance
                </h4>
            """, unsafe_allow_html=True)

            fig = px.bar(
                importance_df.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Blues'
            )

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#1e3a5f'),
                showlegend=False,
                yaxis={'categoryorder': 'total ascending'},
                margin=dict(l=20, r=20, t=20, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

    # Reference Information
    with st.expander("ℹ️ Understanding the Parameters"):
        st.markdown("""
            **Parameter Descriptions:**

            | Parameter | Description | Normal Range |
            |-----------|-------------|--------------|
            | **Pregnancies** | Number of times pregnant | 0-17 |
            | **Glucose** | Plasma glucose concentration (mg/dL) | 70-100 (fasting) |
            | **Blood Pressure** | Diastolic blood pressure (mm Hg) | 60-80 |
            | **Skin Thickness** | Triceps skin fold thickness (mm) | 10-50 |
            | **Insulin** | 2-Hour serum insulin (μU/mL) | 16-166 |
            | **BMI** | Body Mass Index (kg/m²) | 18.5-24.9 |
            | **Diabetes Pedigree** | Genetic influence score | 0.078-2.42 |
            | **Age** | Age in years | - |

            **Risk Factors for Type 2 Diabetes:**
            - Overweight or obesity (BMI ≥ 25)
            - Age 45 or older
            - Family history of diabetes
            - Physical inactivity
            - High blood pressure
            - Abnormal cholesterol levels
        """)

    # Disclaimer
    render_disclaimer()
