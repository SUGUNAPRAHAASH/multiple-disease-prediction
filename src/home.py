"""
Home page for the Multi-Disease Prediction System.
Displays dashboard overview and navigation cards.
"""

import sys
from pathlib import Path
# Add project root to path for Streamlit Cloud compatibility
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.styles import render_header, render_disclaimer


def render_home_page():
    """Render the home page with dashboard overview."""

    # Header
    render_header(
        title="Welcome to HealthPredict AI",
        subtitle="Advanced Machine Learning for Early Disease Detection",
        icon="🏥"
    )

    # Hero Section
    st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3a5f 0%, #3b82f6 100%);
                    border-radius: 20px; padding: 2.5rem; margin-bottom: 2rem;
                    box-shadow: 0 20px 25px -5px rgba(30, 58, 95, 0.2);">
            <h2 style="color: white; margin-bottom: 1rem; font-size: 1.8rem;">
                Your Personal Health Risk Assessment Platform
            </h2>
            <p style="color: #bfdbfe; font-size: 1.1rem; line-height: 1.7; margin-bottom: 1.5rem;">
                Powered by advanced machine learning algorithms, our system analyzes your health
                parameters to provide instant risk assessments for multiple diseases. Get insights
                that can help guide discussions with your healthcare provider.
            </p>
            <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem;
                            border-radius: 50px; color: white; font-size: 0.9rem;">
                    ✓ Instant Results
                </span>
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem;
                            border-radius: 50px; color: white; font-size: 0.9rem;">
                    ✓ ML-Powered Analysis
                </span>
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem;
                            border-radius: 50px; color: white; font-size: 0.9rem;">
                    ✓ Privacy-First
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Disease Cards Section
    st.markdown("""
        <h3 style="color: #1e3a5f; margin-bottom: 1.5rem;">
            Available Health Assessments
        </h3>
    """, unsafe_allow_html=True)

    # Create 2x2 grid of disease cards
    col1, col2 = st.columns(2)

    with col1:
        # Diabetes Card
        st.markdown("""
            <div class="health-card" style="margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                    <span style="font-size: 2.5rem;">🩸</span>
                    <div>
                        <h4 style="color: #1e3a5f; margin: 0;">Diabetes Risk Assessment</h4>
                        <p style="color: #64748b; margin: 0; font-size: 0.85rem;">PIMA Indians Dataset</p>
                    </div>
                </div>
                <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                    Evaluate your risk of Type 2 Diabetes based on factors like glucose levels,
                    BMI, blood pressure, and family history. Early detection is key to prevention.
                </p>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
                    <span style="color: #3b82f6; font-size: 0.85rem; font-weight: 500;">
                        8 Health Parameters Analyzed
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Parkinson's Card
        st.markdown("""
            <div class="health-card">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                    <span style="font-size: 2.5rem;">🧠</span>
                    <div>
                        <h4 style="color: #1e3a5f; margin: 0;">Parkinson's Screening</h4>
                        <p style="color: #64748b; margin: 0; font-size: 0.85rem;">Voice Analysis Dataset</p>
                    </div>
                </div>
                <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                    Voice-based screening for Parkinson's disease using biomedical voice measurements.
                    Analyzes jitter, shimmer, and other acoustic features.
                </p>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
                    <span style="color: #3b82f6; font-size: 0.85rem; font-weight: 500;">
                        22 Voice Parameters Analyzed
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        # Heart Disease Card
        st.markdown("""
            <div class="health-card" style="margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                    <span style="font-size: 2.5rem;">❤️</span>
                    <div>
                        <h4 style="color: #1e3a5f; margin: 0;">Heart Disease Check</h4>
                        <p style="color: #64748b; margin: 0; font-size: 0.85rem;">UCI Heart Dataset</p>
                    </div>
                </div>
                <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                    Comprehensive cardiac risk assessment analyzing cholesterol, blood pressure,
                    ECG results, and other cardiovascular indicators.
                </p>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
                    <span style="color: #3b82f6; font-size: 0.85rem; font-weight: 500;">
                        13 Cardiac Parameters Analyzed
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Liver Disease Card
        st.markdown("""
            <div class="health-card">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                    <span style="font-size: 2.5rem;">🫁</span>
                    <div>
                        <h4 style="color: #1e3a5f; margin: 0;">Liver Health Analysis</h4>
                        <p style="color: #64748b; margin: 0; font-size: 0.85rem;">Indian Liver Patient Dataset</p>
                    </div>
                </div>
                <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
                    Liver function assessment based on bilirubin levels, enzyme markers,
                    protein levels, and other hepatic indicators.
                </p>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
                    <span style="color: #3b82f6; font-size: 0.85rem; font-weight: 500;">
                        10 Liver Parameters Analyzed
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # How It Works Section
    st.markdown("---")
    st.markdown("""
        <h3 style="color: #1e3a5f; margin-bottom: 1.5rem; text-align: center;">
            How It Works
        </h3>
    """, unsafe_allow_html=True)

    cols = st.columns(4)

    steps = [
        ("1", "Select Disease", "Choose the health assessment you want to take from the sidebar", "🎯"),
        ("2", "Enter Data", "Input your health parameters in the easy-to-use form", "📝"),
        ("3", "Get Analysis", "Our ML models analyze your data instantly", "⚡"),
        ("4", "Review Results", "Receive your risk assessment with explanations", "📊")
    ]

    for col, (num, title, desc, icon) in zip(cols, steps):
        with col:
            st.markdown(f"""
                <div style="text-align: center; padding: 1rem;">
                    <div style="background: linear-gradient(135deg, #3b82f6 0%, #1e3a5f 100%);
                                width: 60px; height: 60px; border-radius: 50%;
                                display: flex; align-items: center; justify-content: center;
                                margin: 0 auto 1rem auto;
                                box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);">
                        <span style="color: white; font-size: 1.5rem; font-weight: 700;">{num}</span>
                    </div>
                    <h4 style="color: #1e3a5f; margin-bottom: 0.5rem;">{title}</h4>
                    <p style="color: #64748b; font-size: 0.85rem; line-height: 1.5;">{desc}</p>
                </div>
            """, unsafe_allow_html=True)

    # Stats Section
    st.markdown("---")

    stat_cols = st.columns(4)
    stats = [
        ("4", "Disease Models", "🔬"),
        ("53+", "Health Parameters", "📊"),
        ("95%+", "Model Accuracy", "✅"),
        ("Instant", "Results", "⚡")
    ]

    for col, (value, label, icon) in zip(stat_cols, stats):
        with col:
            st.markdown(f"""
                <div style="background: white; padding: 1.5rem; border-radius: 12px;
                            text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <span style="font-size: 2rem;">{icon}</span>
                    <h3 style="color: #1e3a5f; margin: 0.5rem 0 0.25rem 0;">{value}</h3>
                    <p style="color: #64748b; margin: 0; font-size: 0.85rem;">{label}</p>
                </div>
            """, unsafe_allow_html=True)

    # Important Notice
    st.markdown("---")
    render_disclaimer()

    # Technology Stack
    with st.expander("🔧 Technology Stack"):
        st.markdown("""
            **Machine Learning Models:**
            - Logistic Regression
            - Random Forest Classifier
            - Gradient Boosting Classifier
            - Support Vector Machine (SVM)

            **Data Processing:**
            - Pandas for data manipulation
            - NumPy for numerical operations
            - Scikit-learn for ML pipelines

            **Frontend:**
            - Streamlit for interactive UI
            - Custom CSS for professional styling

            **Model Selection:**
            - Automatic model comparison using cross-validation
            - Best model selection based on accuracy and generalization
        """)
