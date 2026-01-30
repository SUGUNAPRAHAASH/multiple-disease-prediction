"""
Heart Disease Prediction Module for Multi-Disease Prediction System.

This module handles:
- Heart disease risk prediction using UCI Heart Disease dataset
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
    train_heart_model,
    make_prediction,
    get_risk_level,
    get_feature_importance
)
from src.styles import render_header, render_disclaimer, render_result_card


def get_heart_explanation(prediction: int, probability: float, inputs: dict) -> str:
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
        explanation = "The analysis indicates potential cardiovascular risk factors in your profile. "

        risk_factors = []
        if inputs['cholesterol'] > 240:
            risk_factors.append("high cholesterol levels")
        if inputs['bp'] > 140:
            risk_factors.append("elevated blood pressure")
        if inputs['max_hr'] < 100:
            risk_factors.append("lower than expected maximum heart rate")
        if inputs['st_depression'] > 2:
            risk_factors.append("significant ST depression")
        if inputs['age'] > 55:
            risk_factors.append("age-related cardiac risk")
        if inputs['chest_pain'] == 4:
            risk_factors.append("asymptomatic chest pain type")

        if risk_factors:
            explanation += f"Contributing factors may include: {', '.join(risk_factors)}. "

        explanation += "Please consult a cardiologist for comprehensive cardiac evaluation and appropriate diagnostic tests."
    else:
        explanation = "Your cardiovascular parameters appear within normal ranges. "
        explanation += "Continue maintaining heart-healthy habits: regular exercise, balanced diet low in saturated fats, stress management, and regular health screenings."

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
        go.Bar(name='Test Accuracy', x=model_names, y=accuracies, marker_color='#ef4444'),
        go.Bar(name='CV Score', x=model_names, y=cv_scores, marker_color='#f97316')
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


def render_heart_page():
    """Render the heart disease prediction page."""

    # Header
    render_header(
        title="Heart Disease Risk Assessment",
        subtitle="Comprehensive cardiac health evaluation using the UCI Heart Disease dataset",
        icon="❤️"
    )

    # Load/train model with spinner
    with st.spinner("Loading heart disease prediction model..."):
        model, scaler, model_name, accuracy, results, feature_names = train_heart_model()

    # Main content layout
    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("""
            <div class="health-card">
                <h3 style="color: #1e3a5f; margin-bottom: 1.5rem;">
                    📋 Enter Your Cardiac Parameters
                </h3>
            </div>
        """, unsafe_allow_html=True)

        with st.form("heart_form"):
            # Row 1
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input(
                    "Age (years)",
                    min_value=1,
                    max_value=120,
                    value=50,
                    help="Age in years"
                )
            with col2:
                sex = st.selectbox(
                    "Sex",
                    options=[1, 0],
                    format_func=lambda x: "Male" if x == 1 else "Female",
                    help="Biological sex"
                )

            # Row 2
            col3, col4 = st.columns(2)
            with col3:
                chest_pain = st.selectbox(
                    "Chest Pain Type",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: {
                        1: "Typical Angina",
                        2: "Atypical Angina",
                        3: "Non-anginal Pain",
                        4: "Asymptomatic"
                    }[x],
                    help="Type of chest pain experienced"
                )
            with col4:
                bp = st.number_input(
                    "Resting Blood Pressure (mm Hg)",
                    min_value=80,
                    max_value=220,
                    value=120,
                    help="Resting blood pressure on admission"
                )

            # Row 3
            col5, col6 = st.columns(2)
            with col5:
                cholesterol = st.number_input(
                    "Cholesterol (mg/dL)",
                    min_value=100,
                    max_value=600,
                    value=200,
                    help="Serum cholesterol level"
                )
            with col6:
                fbs = st.selectbox(
                    "Fasting Blood Sugar > 120 mg/dL",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    help="Fasting blood sugar greater than 120 mg/dL"
                )

            # Row 4
            col7, col8 = st.columns(2)
            with col7:
                ekg = st.selectbox(
                    "Resting EKG Results",
                    options=[0, 1, 2],
                    format_func=lambda x: {
                        0: "Normal",
                        1: "ST-T Abnormality",
                        2: "Left Ventricular Hypertrophy"
                    }[x],
                    help="Resting electrocardiographic results"
                )
            with col8:
                max_hr = st.number_input(
                    "Maximum Heart Rate",
                    min_value=60,
                    max_value=220,
                    value=150,
                    help="Maximum heart rate achieved during exercise"
                )

            # Row 5
            col9, col10 = st.columns(2)
            with col9:
                exercise_angina = st.selectbox(
                    "Exercise Induced Angina",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    help="Exercise induced angina"
                )
            with col10:
                st_depression = st.number_input(
                    "ST Depression",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    help="ST depression induced by exercise relative to rest"
                )

            # Row 6
            col11, col12 = st.columns(2)
            with col11:
                slope = st.selectbox(
                    "Slope of Peak Exercise ST",
                    options=[1, 2, 3],
                    format_func=lambda x: {
                        1: "Upsloping",
                        2: "Flat",
                        3: "Downsloping"
                    }[x],
                    help="Slope of the peak exercise ST segment"
                )
            with col12:
                vessels = st.selectbox(
                    "Number of Major Vessels",
                    options=[0, 1, 2, 3],
                    help="Number of major vessels colored by fluoroscopy (0-3)"
                )

            # Row 7
            thallium = st.selectbox(
                "Thallium Stress Test",
                options=[3, 6, 7],
                format_func=lambda x: {
                    3: "Normal",
                    6: "Fixed Defect",
                    7: "Reversible Defect"
                }[x],
                help="Thallium stress test result"
            )

            # Submit button
            submitted = st.form_submit_button(
                "🔍 Analyze Cardiac Risk",
                use_container_width=True,
                type="primary"
            )

    with col_result:
        if submitted:
            # Prepare input features (matching dataset column order)
            features = np.array([
                age, sex, chest_pain, bp, cholesterol, fbs, ekg,
                max_hr, exercise_angina, st_depression, slope, vessels, thallium
            ])

            # Make prediction
            prediction, probability = make_prediction(model, scaler, features)

            # Determine risk level
            risk_level = get_risk_level(probability, prediction)

            # Get explanation
            inputs = {
                'cholesterol': cholesterol,
                'bp': bp,
                'max_hr': max_hr,
                'st_depression': st_depression,
                'age': age,
                'chest_pain': chest_pain
            }
            explanation = get_heart_explanation(prediction, probability, inputs)

            # Display result
            render_result_card(
                prediction=prediction,
                disease_name="Heart Disease",
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
                'Parameter': ['Age', 'Sex', 'Chest Pain Type', 'Blood Pressure', 'Cholesterol',
                             'Fasting Blood Sugar', 'EKG Results', 'Max Heart Rate',
                             'Exercise Angina', 'ST Depression', 'ST Slope', 'Vessels', 'Thallium'],
                'Value': [f"{age} years", "Male" if sex == 1 else "Female",
                         {1: "Typical Angina", 2: "Atypical", 3: "Non-anginal", 4: "Asymptomatic"}[chest_pain],
                         f"{bp} mm Hg", f"{cholesterol} mg/dL",
                         "Yes" if fbs == 1 else "No",
                         {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[ekg],
                         f"{max_hr} bpm", "Yes" if exercise_angina == 1 else "No",
                         f"{st_depression}", {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[slope],
                         str(vessels), {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[thallium]]
            })

            st.dataframe(input_df, use_container_width=True, hide_index=True)

        else:
            # Show placeholder when no prediction yet
            st.markdown("""
                <div style="background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
                            border-radius: 16px; padding: 3rem; text-align: center;
                            border: 2px dashed #fca5a5;">
                    <span style="font-size: 4rem;">❤️</span>
                    <h3 style="color: #1e3a5f; margin: 1rem 0;">Ready for Cardiac Assessment</h3>
                    <p style="color: #64748b;">
                        Enter your cardiac parameters on the left and click
                        <strong>"Analyze Cardiac Risk"</strong> to get your heart disease risk assessment.
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
                color_continuous_scale='Reds'
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

            | Parameter | Description | Significance |
            |-----------|-------------|--------------|
            | **Age** | Patient's age in years | Risk increases with age |
            | **Sex** | Male/Female | Males have higher risk |
            | **Chest Pain Type** | Type of chest discomfort | Asymptomatic is concerning |
            | **Blood Pressure** | Resting BP (mm Hg) | Normal: <120/80 |
            | **Cholesterol** | Serum cholesterol (mg/dL) | Desirable: <200 |
            | **Fasting Blood Sugar** | FBS > 120 mg/dL | Indicates diabetes risk |
            | **EKG Results** | Resting ECG findings | Abnormalities increase risk |
            | **Max Heart Rate** | Maximum HR during exercise | Lower values concerning |
            | **Exercise Angina** | Chest pain during exercise | Indicates coronary issues |
            | **ST Depression** | Exercise-induced ST changes | Higher values = more risk |
            | **ST Slope** | Peak exercise ST segment | Downsloping is concerning |
            | **Vessels** | Major vessels with blockage | More vessels = higher risk |
            | **Thallium** | Nuclear stress test result | Defects indicate damage |

            **Heart Disease Risk Factors:**
            - High blood pressure (≥140/90 mm Hg)
            - High cholesterol (≥240 mg/dL)
            - Smoking and tobacco use
            - Diabetes
            - Obesity and physical inactivity
            - Family history of heart disease
        """)

    # Disclaimer
    render_disclaimer()
