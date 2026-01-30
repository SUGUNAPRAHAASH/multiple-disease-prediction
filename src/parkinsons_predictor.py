"""
Parkinson's Disease Prediction Module for Multi-Disease Prediction System.

This module handles:
- Parkinson's disease risk prediction using voice measurements
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
    train_parkinsons_model,
    make_prediction,
    get_risk_level,
    get_feature_importance
)
from src.styles import render_header, render_disclaimer, render_result_card


def get_parkinsons_explanation(prediction: int, probability: float, inputs: dict) -> str:
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
        explanation = "The voice analysis indicates patterns that may be associated with Parkinson's disease. "

        risk_factors = []
        if inputs.get('jitter', 0) > 0.01:
            risk_factors.append("elevated voice jitter")
        if inputs.get('shimmer', 0) > 0.05:
            risk_factors.append("increased voice shimmer")
        if inputs.get('nhr', 0) > 0.03:
            risk_factors.append("higher noise-to-harmonics ratio")
        if inputs.get('hnr', 25) < 20:
            risk_factors.append("reduced harmonics-to-noise ratio")
        if inputs.get('spread1', 0) > -5:
            risk_factors.append("voice frequency spread patterns")

        if risk_factors:
            explanation += f"Notable voice characteristics include: {', '.join(risk_factors)}. "

        explanation += "This screening tool uses voice biomarkers. Please consult a neurologist for comprehensive evaluation including motor symptoms assessment."
    else:
        explanation = "Your voice parameters fall within ranges typically associated with healthy individuals. "
        explanation += "However, this is a screening tool and does not replace clinical diagnosis. If you have concerns about neurological symptoms, please consult a healthcare provider."

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
        go.Bar(name='Test Accuracy', x=model_names, y=accuracies, marker_color='#8b5cf6'),
        go.Bar(name='CV Score', x=model_names, y=cv_scores, marker_color='#a78bfa')
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


def render_parkinsons_page():
    """Render the Parkinson's disease prediction page."""

    # Header
    render_header(
        title="Parkinson's Disease Screening",
        subtitle="Voice-based screening using biomedical voice measurements",
        icon="🧠"
    )

    # Load/train model with spinner
    with st.spinner("Loading Parkinson's disease prediction model..."):
        model, scaler, model_name, accuracy, results, feature_names = train_parkinsons_model()

    # Information banner about the test
    st.markdown("""
        <div style="background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
                    border-left: 4px solid #8b5cf6; border-radius: 0 12px 12px 0;
                    padding: 1rem 1.5rem; margin-bottom: 2rem;">
            <p style="color: #5b21b6; margin: 0; font-size: 0.95rem;">
                <strong>ℹ️ About This Test:</strong> This screening uses voice measurements to detect
                potential Parkinson's disease. The analysis is based on various voice frequency and
                amplitude measurements. For accurate input, voice measurements should be obtained
                from specialized equipment or voice analysis software.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Main content layout
    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("""
            <div class="health-card">
                <h3 style="color: #1e3a5f; margin-bottom: 1.5rem;">
                    📋 Enter Voice Measurements
                </h3>
            </div>
        """, unsafe_allow_html=True)

        # Use tabs to organize the many parameters
        tab1, tab2, tab3 = st.tabs(["📊 Frequency Measures", "📈 Amplitude Measures", "🔬 Other Measures"])

        with st.form("parkinsons_form"):
            with tab1:
                st.markdown("**Fundamental Frequency Parameters:**")
                col1, col2 = st.columns(2)
                with col1:
                    fo = st.number_input(
                        "MDVP:Fo (Hz)",
                        min_value=80.0,
                        max_value=300.0,
                        value=150.0,
                        step=0.1,
                        help="Average vocal fundamental frequency"
                    )
                    flo = st.number_input(
                        "MDVP:Flo (Hz)",
                        min_value=50.0,
                        max_value=250.0,
                        value=100.0,
                        step=0.1,
                        help="Minimum vocal fundamental frequency"
                    )
                with col2:
                    fhi = st.number_input(
                        "MDVP:Fhi (Hz)",
                        min_value=100.0,
                        max_value=600.0,
                        value=200.0,
                        step=0.1,
                        help="Maximum vocal fundamental frequency"
                    )

                st.markdown("**Jitter Parameters (Frequency Variation):**")
                col3, col4 = st.columns(2)
                with col3:
                    jitter_percent = st.number_input(
                        "MDVP:Jitter (%)",
                        min_value=0.0,
                        max_value=0.1,
                        value=0.005,
                        step=0.0001,
                        format="%.5f",
                        help="Jitter as percentage"
                    )
                    jitter_rap = st.number_input(
                        "MDVP:RAP",
                        min_value=0.0,
                        max_value=0.05,
                        value=0.003,
                        step=0.0001,
                        format="%.5f",
                        help="Relative amplitude perturbation"
                    )
                    jitter_ddp = st.number_input(
                        "Jitter:DDP",
                        min_value=0.0,
                        max_value=0.1,
                        value=0.01,
                        step=0.0001,
                        format="%.5f",
                        help="Average absolute difference of differences between consecutive periods"
                    )
                with col4:
                    jitter_abs = st.number_input(
                        "MDVP:Jitter (Abs)",
                        min_value=0.0,
                        max_value=0.001,
                        value=0.00005,
                        step=0.000001,
                        format="%.6f",
                        help="Absolute jitter in microseconds"
                    )
                    jitter_ppq = st.number_input(
                        "MDVP:PPQ",
                        min_value=0.0,
                        max_value=0.05,
                        value=0.003,
                        step=0.0001,
                        format="%.5f",
                        help="Five-point period perturbation quotient"
                    )

            with tab2:
                st.markdown("**Shimmer Parameters (Amplitude Variation):**")
                col5, col6 = st.columns(2)
                with col5:
                    shimmer = st.number_input(
                        "MDVP:Shimmer",
                        min_value=0.0,
                        max_value=0.2,
                        value=0.03,
                        step=0.001,
                        format="%.4f",
                        help="Local shimmer"
                    )
                    shimmer_apq3 = st.number_input(
                        "Shimmer:APQ3",
                        min_value=0.0,
                        max_value=0.1,
                        value=0.015,
                        step=0.001,
                        format="%.4f",
                        help="Three-point amplitude perturbation quotient"
                    )
                    shimmer_apq = st.number_input(
                        "MDVP:APQ",
                        min_value=0.0,
                        max_value=0.1,
                        value=0.02,
                        step=0.001,
                        format="%.4f",
                        help="11-point amplitude perturbation quotient"
                    )
                with col6:
                    shimmer_db = st.number_input(
                        "MDVP:Shimmer (dB)",
                        min_value=0.0,
                        max_value=2.0,
                        value=0.3,
                        step=0.01,
                        format="%.3f",
                        help="Local shimmer in decibels"
                    )
                    shimmer_apq5 = st.number_input(
                        "Shimmer:APQ5",
                        min_value=0.0,
                        max_value=0.1,
                        value=0.02,
                        step=0.001,
                        format="%.4f",
                        help="Five-point amplitude perturbation quotient"
                    )
                    shimmer_dda = st.number_input(
                        "Shimmer:DDA",
                        min_value=0.0,
                        max_value=0.2,
                        value=0.05,
                        step=0.001,
                        format="%.4f",
                        help="Average absolute difference between amplitudes"
                    )

            with tab3:
                st.markdown("**Noise & Harmonicity:**")
                col7, col8 = st.columns(2)
                with col7:
                    nhr = st.number_input(
                        "NHR",
                        min_value=0.0,
                        max_value=0.5,
                        value=0.02,
                        step=0.001,
                        format="%.4f",
                        help="Noise-to-harmonics ratio"
                    )
                with col8:
                    hnr = st.number_input(
                        "HNR",
                        min_value=0.0,
                        max_value=40.0,
                        value=22.0,
                        step=0.1,
                        help="Harmonics-to-noise ratio"
                    )

                st.markdown("**Nonlinear Measures:**")
                col9, col10 = st.columns(2)
                with col9:
                    rpde = st.number_input(
                        "RPDE",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.01,
                        format="%.4f",
                        help="Recurrence period density entropy"
                    )
                    spread1 = st.number_input(
                        "Spread1",
                        min_value=-10.0,
                        max_value=0.0,
                        value=-5.0,
                        step=0.1,
                        format="%.4f",
                        help="Nonlinear measure of fundamental frequency variation"
                    )
                    d2 = st.number_input(
                        "D2",
                        min_value=1.0,
                        max_value=4.0,
                        value=2.5,
                        step=0.01,
                        format="%.4f",
                        help="Correlation dimension"
                    )
                with col10:
                    dfa = st.number_input(
                        "DFA",
                        min_value=0.5,
                        max_value=1.0,
                        value=0.7,
                        step=0.01,
                        format="%.4f",
                        help="Detrended fluctuation analysis"
                    )
                    spread2 = st.number_input(
                        "Spread2",
                        min_value=0.0,
                        max_value=0.5,
                        value=0.25,
                        step=0.01,
                        format="%.4f",
                        help="Nonlinear measure of fundamental frequency variation"
                    )
                    ppe = st.number_input(
                        "PPE",
                        min_value=0.0,
                        max_value=0.6,
                        value=0.2,
                        step=0.01,
                        format="%.4f",
                        help="Pitch period entropy"
                    )

            # Submit button
            submitted = st.form_submit_button(
                "🔍 Analyze Voice Patterns",
                use_container_width=True,
                type="primary"
            )

    with col_result:
        if submitted:
            # Prepare input features (matching dataset column order after dropping 'name')
            features = np.array([
                fo, fhi, flo, jitter_percent, jitter_abs, jitter_rap, jitter_ppq,
                jitter_ddp, shimmer, shimmer_db, shimmer_apq3, shimmer_apq5,
                shimmer_apq, shimmer_dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe
            ])

            # Make prediction
            prediction, probability = make_prediction(model, scaler, features)

            # Determine risk level
            risk_level = get_risk_level(probability, prediction)

            # Get explanation
            inputs = {
                'jitter': jitter_percent,
                'shimmer': shimmer,
                'nhr': nhr,
                'hnr': hnr,
                'spread1': spread1
            }
            explanation = get_parkinsons_explanation(prediction, probability, inputs)

            # Display result
            render_result_card(
                prediction=prediction,
                disease_name="Parkinson's Disease",
                probability=probability,
                risk_level=risk_level,
                explanation=explanation
            )

            # Key metrics visualization
            st.markdown("---")
            st.markdown("""
                <h4 style="color: #1e3a5f; margin: 1rem 0;">
                    📊 Key Voice Metrics
                </h4>
            """, unsafe_allow_html=True)

            # Create a radar chart for key metrics
            categories = ['Jitter', 'Shimmer', 'NHR', 'HNR', 'RPDE', 'PPE']
            # Normalize values for visualization
            values = [
                min(jitter_percent / 0.02, 1),  # Normalize jitter
                min(shimmer / 0.1, 1),  # Normalize shimmer
                min(nhr / 0.1, 1),  # Normalize NHR
                hnr / 30,  # Normalize HNR
                rpde,  # Already 0-1
                min(ppe / 0.4, 1)  # Normalize PPE
            ]

            fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                fillcolor='rgba(139, 92, 246, 0.3)',
                line=dict(color='#8b5cf6', width=2)
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickvals=[0.25, 0.5, 0.75, 1],
                        ticktext=['Low', 'Medium', 'High', 'Very High']
                    )
                ),
                showlegend=False,
                margin=dict(l=60, r=60, t=40, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            # Show placeholder when no prediction yet
            st.markdown("""
                <div style="background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
                            border-radius: 16px; padding: 3rem; text-align: center;
                            border: 2px dashed #c4b5fd;">
                    <span style="font-size: 4rem;">🧠</span>
                    <h3 style="color: #1e3a5f; margin: 1rem 0;">Ready for Voice Analysis</h3>
                    <p style="color: #64748b;">
                        Enter your voice measurement parameters using the tabs on the left
                        and click <strong>"Analyze Voice Patterns"</strong> to get your
                        Parkinson's disease screening result.
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
                color_continuous_scale='Purples'
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
    with st.expander("ℹ️ Understanding Voice Biomarkers"):
        st.markdown("""
            **Voice Biomarker Categories:**

            **Jitter (Frequency Variation):**
            - Measures cycle-to-cycle variations in fundamental frequency
            - Higher jitter values may indicate voice instability
            - Parameters: Jitter(%), Jitter(Abs), RAP, PPQ, DDP

            **Shimmer (Amplitude Variation):**
            - Measures cycle-to-cycle variations in amplitude
            - Higher shimmer values may indicate voice amplitude instability
            - Parameters: Shimmer, Shimmer(dB), APQ3, APQ5, APQ, DDA

            **Noise Measures:**
            - NHR (Noise-to-Harmonics Ratio): Higher values indicate more noise
            - HNR (Harmonics-to-Noise Ratio): Lower values indicate more noise

            **Nonlinear Measures:**
            - RPDE: Recurrence period density entropy
            - DFA: Detrended fluctuation analysis
            - Spread1, Spread2: Frequency variation measures
            - D2: Correlation dimension
            - PPE: Pitch period entropy

            **About Parkinson's Disease:**
            - Progressive neurological disorder affecting movement
            - Voice changes can be an early symptom
            - Motor symptoms include tremor, rigidity, bradykinesia
            - Early detection allows for better management
        """)

    # Disclaimer
    render_disclaimer()
