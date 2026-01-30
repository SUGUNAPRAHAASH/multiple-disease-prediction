"""
Custom CSS styles for the Multi-Disease Prediction System.
Provides a professional healthcare-themed UI design.
"""

import streamlit as st


def apply_custom_styles():
    """Apply custom CSS styles to transform the Streamlit UI into a professional healthcare interface."""

    st.markdown("""
        <style>
        /* ============================================
           GLOBAL STYLES & VARIABLES
           ============================================ */

        :root {
            --primary-blue: #1e3a5f;
            --primary-light: #3b82f6;
            --accent-teal: #0d9488;
            --accent-green: #059669;
            --background-light: #f8fafc;
            --card-bg: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border-color: #e2e8f0;
            --success-green: #10b981;
            --warning-yellow: #f59e0b;
            --danger-red: #ef4444;
            --info-blue: #3b82f6;
        }

        /* Main container background */
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* ============================================
           SIDEBAR STYLES
           ============================================ */

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border-right: 1px solid #e2e8f0;
        }

        [data-testid="stSidebar"] [data-testid="stButton"] button {
            width: 100%;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            font-weight: 500;
            transition: all 0.2s ease;
            text-align: left;
            justify-content: flex-start;
        }

        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
            color: white;
            border: none;
            box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
        }

        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] {
            background: transparent;
            color: #475569;
            border: 1px solid #e2e8f0;
        }

        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]:hover {
            background: #f1f5f9;
            border-color: #3b82f6;
            color: #1e3a5f;
        }

        /* ============================================
           MAIN CONTENT STYLES
           ============================================ */

        .main .block-container {
            padding: 2rem 3rem;
            max-width: 1400px;
        }

        /* Headers */
        h1 {
            color: #1e3a5f !important;
            font-weight: 700 !important;
            letter-spacing: -0.025em;
        }

        h2 {
            color: #334155 !important;
            font-weight: 600 !important;
        }

        h3 {
            color: #475569 !important;
            font-weight: 600 !important;
        }

        /* ============================================
           CARD COMPONENTS
           ============================================ */

        .health-card {
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
            border: 1px solid #e2e8f0;
            transition: all 0.3s ease;
        }

        .health-card:hover {
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
            transform: translateY(-2px);
        }

        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 12px;
            padding: 1.25rem;
            border-left: 4px solid #3b82f6;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .metric-card.success {
            border-left-color: #10b981;
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        }

        .metric-card.warning {
            border-left-color: #f59e0b;
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        }

        .metric-card.danger {
            border-left-color: #ef4444;
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        }

        /* ============================================
           FORM ELEMENTS
           ============================================ */

        /* Text inputs and number inputs */
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input {
            border-radius: 10px !important;
            border: 2px solid #e2e8f0 !important;
            padding: 0.75rem 1rem !important;
            font-size: 1rem !important;
            transition: all 0.2s ease !important;
        }

        [data-testid="stNumberInput"] input:focus,
        [data-testid="stTextInput"] input:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
        }

        /* Select boxes */
        [data-testid="stSelectbox"] > div > div {
            border-radius: 10px !important;
            border: 2px solid #e2e8f0 !important;
        }

        /* Sliders */
        [data-testid="stSlider"] > div > div > div {
            background: #3b82f6 !important;
        }

        /* ============================================
           BUTTONS
           ============================================ */

        .stButton > button {
            border-radius: 10px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
            border: none;
            box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3);
        }

        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #2563eb 0%, #1e3a5f 100%);
            box-shadow: 0 6px 10px -1px rgba(37, 99, 235, 0.4);
            transform: translateY(-1px);
        }

        /* ============================================
           RESULT DISPLAYS
           ============================================ */

        .result-positive {
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border: 2px solid #fca5a5;
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
        }

        .result-negative {
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            border: 2px solid #6ee7b7;
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
        }

        .result-warning {
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            border: 2px solid #fcd34d;
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
        }

        /* ============================================
           RISK INDICATORS
           ============================================ */

        .risk-badge {
            display: inline-block;
            padding: 0.5rem 1.5rem;
            border-radius: 50px;
            font-weight: 700;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .risk-low {
            background: #10b981;
            color: white;
        }

        .risk-medium {
            background: #f59e0b;
            color: white;
        }

        .risk-high {
            background: #ef4444;
            color: white;
        }

        /* ============================================
           METRICS & STATS
           ============================================ */

        [data-testid="stMetric"] {
            background: white;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        [data-testid="stMetric"] label {
            color: #64748b !important;
            font-weight: 500;
        }

        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #1e3a5f !important;
            font-weight: 700;
        }

        /* ============================================
           EXPANDERS
           ============================================ */

        [data-testid="stExpander"] {
            background: white;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }

        [data-testid="stExpander"] summary {
            font-weight: 600;
            color: #334155;
        }

        /* ============================================
           TABS
           ============================================ */

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: transparent;
        }

        .stTabs [data-baseweb="tab"] {
            background: white;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
            color: white;
            border: none;
        }

        /* ============================================
           ALERTS & NOTICES
           ============================================ */

        .stAlert {
            border-radius: 12px;
        }

        /* Medical Disclaimer */
        .disclaimer {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border: 1px solid #f59e0b;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
        }

        .disclaimer p {
            color: #92400e;
            font-size: 0.875rem;
            margin: 0;
        }

        /* ============================================
           PROGRESS INDICATORS
           ============================================ */

        .stProgress > div > div > div {
            background: linear-gradient(90deg, #3b82f6 0%, #10b981 100%);
            border-radius: 10px;
        }

        /* ============================================
           CHARTS & VISUALIZATIONS
           ============================================ */

        [data-testid="stPlotlyChart"] {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        /* ============================================
           RESPONSIVE DESIGN
           ============================================ */

        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem;
            }

            .health-card {
                padding: 1rem;
            }
        }

        /* ============================================
           ANIMATIONS
           ============================================ */

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .animate-fade-in {
            animation: fadeIn 0.5s ease-out forwards;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        .animate-pulse {
            animation: pulse 2s ease-in-out infinite;
        }

        /* ============================================
           CUSTOM SCROLLBAR
           ============================================ */

        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #f1f5f9;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }

        /* ============================================
           LOADING STATES
           ============================================ */

        .stSpinner > div {
            border-color: #3b82f6 !important;
        }

        /* ============================================
           DIVIDERS
           ============================================ */

        hr {
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 2rem 0;
        }
        </style>
    """, unsafe_allow_html=True)


def render_header(title: str, subtitle: str = "", icon: str = ""):
    """Render a styled page header."""
    st.markdown(f"""
        <div style="margin-bottom: 2rem;">
            <h1 style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                <span style="font-size: 2.5rem;">{icon}</span>
                {title}
            </h1>
            <p style="color: #64748b; font-size: 1.1rem; margin: 0;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)


def render_card(content: str, title: str = ""):
    """Render a styled card component."""
    title_html = f'<h3 style="margin-bottom: 1rem; color: #1e3a5f;">{title}</h3>' if title else ""
    st.markdown(f"""
        <div class="health-card">
            {title_html}
            {content}
        </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, status: str = "info"):
    """Render a styled metric card."""
    status_class = {
        "success": "success",
        "warning": "warning",
        "danger": "danger",
        "info": ""
    }.get(status, "")

    st.markdown(f"""
        <div class="metric-card {status_class}">
            <p style="color: #64748b; font-size: 0.875rem; margin-bottom: 0.25rem;">{label}</p>
            <p style="color: #1e3a5f; font-size: 1.5rem; font-weight: 700; margin: 0;">{value}</p>
        </div>
    """, unsafe_allow_html=True)


def render_disclaimer():
    """Render the medical disclaimer."""
    st.markdown("""
        <div class="disclaimer">
            <p>
                <strong>⚠️ Medical Disclaimer:</strong> This application provides informational insights only
                and is not a medical diagnosis. The predictions are based on machine learning models and should
                not replace professional medical advice. Please consult a healthcare provider for proper diagnosis
                and treatment.
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_result_card(prediction: int, disease_name: str, probability: float = None,
                       risk_level: str = "Medium", explanation: str = ""):
    """Render the prediction result in a styled card."""

    if prediction == 1:  # Positive prediction (at risk)
        result_class = "result-positive" if risk_level == "High" else "result-warning"
        icon = "⚠️" if risk_level == "High" else "⚡"
        status_text = f"At Risk for {disease_name}"
        status_color = "#dc2626" if risk_level == "High" else "#d97706"
    else:  # Negative prediction (low risk)
        result_class = "result-negative"
        icon = "✅"
        status_text = f"Low Risk for {disease_name}"
        status_color = "#059669"

    risk_badge_class = {
        "Low": "risk-low",
        "Medium": "risk-medium",
        "High": "risk-high"
    }.get(risk_level, "risk-medium")

    probability_html = ""
    if probability is not None:
        prob_percentage = probability * 100
        probability_html = f"""
            <div style="margin: 1.5rem 0;">
                <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 0.5rem;">Confidence Score</p>
                <div style="background: #e2e8f0; border-radius: 10px; height: 12px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #3b82f6 0%, {'#10b981' if prediction == 0 else '#ef4444'} 100%);
                                width: {prob_percentage}%; height: 100%; border-radius: 10px;
                                transition: width 0.5s ease;"></div>
                </div>
                <p style="color: #1e3a5f; font-weight: 600; margin-top: 0.5rem;">{prob_percentage:.1f}%</p>
            </div>
        """

    explanation_html = f"""
        <p style="color: #475569; font-size: 0.95rem; line-height: 1.6; text-align: left; margin-top: 1.5rem;
                  padding: 1rem; background: rgba(255,255,255,0.7); border-radius: 10px;">
            {explanation}
        </p>
    """ if explanation else ""

    st.markdown(f"""
        <div class="{result_class} animate-fade-in">
            <span style="font-size: 4rem;">{icon}</span>
            <h2 style="color: {status_color}; margin: 1rem 0 0.5rem 0;">{status_text}</h2>
            <span class="risk-badge {risk_badge_class}">{risk_level} Risk</span>
            {probability_html}
            {explanation_html}
        </div>
    """, unsafe_allow_html=True)
