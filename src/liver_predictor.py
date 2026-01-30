"""
Liver Disease Prediction Module for Multi-Disease Prediction System.

This module handles:
- Liver disease risk prediction using Indian Liver Patient Dataset
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
    train_liver_model,
    make_prediction,
    get_risk_level,
    get_feature_importance
)
from src.styles import render_header, render_disclaimer, render_result_card


def get_liver_explanation(prediction: int, probability: float, inputs: dict) -> str:
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
        explanation = "The analysis of your liver function parameters indicates potential concerns. "

        risk_factors = []
        if inputs['total_bilirubin'] > 1.2:
            risk_factors.append("elevated bilirubin levels")
        if inputs['alkaline_phosphotase'] > 150:
            risk_factors.append("elevated alkaline phosphatase")
        if inputs['alt'] > 40:
            risk_factors.append("elevated ALT (SGPT) levels")
        if inputs['ast'] > 40:
            risk_factors.append("elevated AST (SGOT) levels")
        if inputs['albumin'] < 3.5:
            risk_factors.append("low albumin levels")
        if inputs['ag_ratio'] and inputs['ag_ratio'] < 1.0:
            risk_factors.append("altered albumin/globulin ratio")

        if risk_factors:
            explanation += f"Notable findings include: {', '.join(risk_factors)}. "

        explanation += "These values may indicate liver stress or dysfunction. Please consult a hepatologist or gastroenterologist for comprehensive evaluation."
    else:
        explanation = "Your liver function parameters appear within normal ranges. "
        explanation += "Continue maintaining a healthy lifestyle: limit alcohol consumption, maintain healthy weight, avoid unnecessary medications, and get regular check-ups."

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
        go.Bar(name='Test Accuracy', x=model_names, y=accuracies, marker_color='#059669'),
        go.Bar(name='CV Score', x=model_names, y=cv_scores, marker_color='#34d399')
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


def render_liver_page():
    """Render the liver disease prediction page."""

    # Header
    render_header(
        title="Liver Health Analysis",
        subtitle="Comprehensive liver function assessment using the Indian Liver Patient Dataset",
        icon="🫁"
    )

    # Load/train model with spinner
    with st.spinner("Loading liver disease prediction model..."):
        model, scaler, model_name, accuracy, results, feature_names = train_liver_model()

    # Main content layout
    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("""
            <div class="health-card">
                <h3 style="color: #1e3a5f; margin-bottom: 1.5rem;">
                    📋 Enter Your Liver Function Parameters
                </h3>
            </div>
        """, unsafe_allow_html=True)

        with st.form("liver_form"):
            # Row 1: Demographics
            st.markdown("**Demographics:**")
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input(
                    "Age (years)",
                    min_value=1,
                    max_value=120,
                    value=45,
                    help="Age in years"
                )
            with col2:
                gender = st.selectbox(
                    "Gender",
                    options=[1, 0],
                    format_func=lambda x: "Male" if x == 1 else "Female",
                    help="Biological gender"
                )

            st.markdown("---")
            st.markdown("**Bilirubin Levels:**")
            col3, col4 = st.columns(2)
            with col3:
                total_bilirubin = st.number_input(
                    "Total Bilirubin (mg/dL)",
                    min_value=0.0,
                    max_value=80.0,
                    value=1.0,
                    step=0.1,
                    help="Total bilirubin level. Normal: 0.1-1.2 mg/dL"
                )
            with col4:
                direct_bilirubin = st.number_input(
                    "Direct Bilirubin (mg/dL)",
                    min_value=0.0,
                    max_value=20.0,
                    value=0.3,
                    step=0.1,
                    help="Direct (conjugated) bilirubin. Normal: 0.0-0.3 mg/dL"
                )

            st.markdown("---")
            st.markdown("**Enzyme Levels:**")
            col5, col6 = st.columns(2)
            with col5:
                alkaline_phosphotase = st.number_input(
                    "Alkaline Phosphatase (IU/L)",
                    min_value=0,
                    max_value=2500,
                    value=200,
                    help="Alkaline phosphatase level. Normal: 44-147 IU/L"
                )
            with col6:
                alt = st.number_input(
                    "ALT / SGPT (IU/L)",
                    min_value=0,
                    max_value=2000,
                    value=25,
                    help="Alanine Aminotransferase. Normal: 7-56 IU/L"
                )

            col7, col8 = st.columns(2)
            with col7:
                ast = st.number_input(
                    "AST / SGOT (IU/L)",
                    min_value=0,
                    max_value=5000,
                    value=30,
                    help="Aspartate Aminotransferase. Normal: 10-40 IU/L"
                )
            with col8:
                st.empty()  # Placeholder for alignment

            st.markdown("---")
            st.markdown("**Protein Levels:**")
            col9, col10 = st.columns(2)
            with col9:
                total_proteins = st.number_input(
                    "Total Proteins (g/dL)",
                    min_value=0.0,
                    max_value=15.0,
                    value=6.5,
                    step=0.1,
                    help="Total protein level. Normal: 6.0-8.3 g/dL"
                )
            with col10:
                albumin = st.number_input(
                    "Albumin (g/dL)",
                    min_value=0.0,
                    max_value=10.0,
                    value=3.5,
                    step=0.1,
                    help="Albumin level. Normal: 3.4-5.4 g/dL"
                )

            # Albumin/Globulin Ratio
            ag_ratio = st.number_input(
                "Albumin/Globulin Ratio",
                min_value=0.0,
                max_value=3.0,
                value=1.0,
                step=0.01,
                help="A/G Ratio. Normal: 1.0-2.5"
            )

            # Submit button
            submitted = st.form_submit_button(
                "🔍 Analyze Liver Health",
                use_container_width=True,
                type="primary"
            )

    with col_result:
        if submitted:
            # Prepare input features (matching dataset column order)
            features = np.array([
                age, gender, total_bilirubin, direct_bilirubin,
                alkaline_phosphotase, alt, ast, total_proteins, albumin, ag_ratio
            ])

            # Make prediction
            prediction, probability = make_prediction(model, scaler, features)

            # Determine risk level
            risk_level = get_risk_level(probability, prediction)

            # Get explanation
            inputs = {
                'total_bilirubin': total_bilirubin,
                'alkaline_phosphotase': alkaline_phosphotase,
                'alt': alt,
                'ast': ast,
                'albumin': albumin,
                'ag_ratio': ag_ratio
            }
            explanation = get_liver_explanation(prediction, probability, inputs)

            # Display result
            render_result_card(
                prediction=prediction,
                disease_name="Liver Disease",
                probability=probability,
                risk_level=risk_level,
                explanation=explanation
            )

            # Liver Function Panel Visualization
            st.markdown("---")
            st.markdown("""
                <h4 style="color: #1e3a5f; margin: 1rem 0;">
                    📊 Liver Function Panel
                </h4>
            """, unsafe_allow_html=True)

            # Create gauge charts for key metrics
            col_g1, col_g2 = st.columns(2)

            with col_g1:
                # Bilirubin gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=total_bilirubin,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Total Bilirubin (mg/dL)", 'font': {'size': 14}},
                    gauge={
                        'axis': {'range': [0, 5], 'tickwidth': 1},
                        'bar': {'color': "#059669"},
                        'steps': [
                            {'range': [0, 1.2], 'color': "#d1fae5"},
                            {'range': [1.2, 3], 'color': "#fef3c7"},
                            {'range': [3, 5], 'color': "#fee2e2"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 1.2
                        }
                    }
                ))
                fig.update_layout(
                    height=200,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_g2:
                # ALT gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=alt,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "ALT / SGPT (IU/L)", 'font': {'size': 14}},
                    gauge={
                        'axis': {'range': [0, 200], 'tickwidth': 1},
                        'bar': {'color': "#059669"},
                        'steps': [
                            {'range': [0, 56], 'color': "#d1fae5"},
                            {'range': [56, 100], 'color': "#fef3c7"},
                            {'range': [100, 200], 'color': "#fee2e2"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 56
                        }
                    }
                ))
                fig.update_layout(
                    height=200,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

            # Input summary
            st.markdown("""
                <h4 style="color: #1e3a5f; margin: 1rem 0;">
                    📋 Your Input Summary
                </h4>
            """, unsafe_allow_html=True)

            input_df = pd.DataFrame({
                'Parameter': ['Age', 'Gender', 'Total Bilirubin', 'Direct Bilirubin',
                             'Alkaline Phosphatase', 'ALT (SGPT)', 'AST (SGOT)',
                             'Total Proteins', 'Albumin', 'A/G Ratio'],
                'Value': [f"{age} years", "Male" if gender == 1 else "Female",
                         f"{total_bilirubin} mg/dL", f"{direct_bilirubin} mg/dL",
                         f"{alkaline_phosphotase} IU/L", f"{alt} IU/L", f"{ast} IU/L",
                         f"{total_proteins} g/dL", f"{albumin} g/dL", f"{ag_ratio:.2f}"],
                'Normal Range': ['N/A', 'N/A', '0.1-1.2 mg/dL', '0.0-0.3 mg/dL',
                                '44-147 IU/L', '7-56 IU/L', '10-40 IU/L',
                                '6.0-8.3 g/dL', '3.4-5.4 g/dL', '1.0-2.5']
            })

            st.dataframe(input_df, use_container_width=True, hide_index=True)

        else:
            # Show placeholder when no prediction yet
            st.markdown("""
                <div style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
                            border-radius: 16px; padding: 3rem; text-align: center;
                            border: 2px dashed #6ee7b7;">
                    <span style="font-size: 4rem;">🫁</span>
                    <h3 style="color: #1e3a5f; margin: 1rem 0;">Ready for Liver Analysis</h3>
                    <p style="color: #64748b;">
                        Enter your liver function test parameters on the left and click
                        <strong>"Analyze Liver Health"</strong> to get your assessment.
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
                color_continuous_scale='Greens'
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
    with st.expander("ℹ️ Understanding Liver Function Tests"):
        st.markdown("""
            **Liver Function Test Parameters:**

            | Parameter | Description | Normal Range | Significance |
            |-----------|-------------|--------------|--------------|
            | **Total Bilirubin** | Waste product from red blood cell breakdown | 0.1-1.2 mg/dL | High = jaundice, liver/bile duct issues |
            | **Direct Bilirubin** | Processed bilirubin | 0.0-0.3 mg/dL | High = bile duct obstruction |
            | **Alkaline Phosphatase** | Enzyme found in liver and bone | 44-147 IU/L | High = bile duct blockage, bone disease |
            | **ALT (SGPT)** | Enzyme mainly found in liver | 7-56 IU/L | High = liver cell damage |
            | **AST (SGOT)** | Enzyme found in liver, heart, muscle | 10-40 IU/L | High = liver or muscle damage |
            | **Total Proteins** | Albumin + Globulin | 6.0-8.3 g/dL | Low = liver disease, malnutrition |
            | **Albumin** | Protein made by liver | 3.4-5.4 g/dL | Low = liver disease |
            | **A/G Ratio** | Albumin to Globulin ratio | 1.0-2.5 | Abnormal = liver/kidney disease |

            **Risk Factors for Liver Disease:**
            - Excessive alcohol consumption
            - Obesity and fatty liver
            - Viral hepatitis (A, B, C)
            - Certain medications
            - Autoimmune conditions
            - Genetic disorders
            - Exposure to toxins
        """)

    # Disclaimer
    render_disclaimer()
