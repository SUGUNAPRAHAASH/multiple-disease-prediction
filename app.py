"""
Multi-Disease Prediction System
A professional healthcare prediction application built with Streamlit.

This application provides ML-powered predictions for:
- Diabetes (PIMA Indians Dataset)
- Heart Disease (UCI Dataset)
- Parkinson's Disease
- Liver Disease (Indian Liver Patient Dataset)

Author: AI Healthcare Solutions
Version: 1.0.0
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path for Streamlit Cloud compatibility
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.styles import apply_custom_styles
from src.diabetes_predictor import render_diabetes_page
from src.heart_predictor import render_heart_page
from src.parkinsons_predictor import render_parkinsons_page
from src.liver_predictor import render_liver_page
from src.home import render_home_page


def initialize_session_state():
    """Initialize session state variables for the application."""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home'


def render_sidebar():
    """Render the sidebar navigation with professional styling."""
    with st.sidebar:
        # Logo/Brand Section
        st.markdown("""
            <div style="text-align: center; padding: 1rem 0 2rem 0;">
                <h1 style="color: #1e3a5f; font-size: 1.5rem; margin-bottom: 0.5rem;">
                    <span style="font-size: 2rem;">🏥</span><br>
                    HealthPredict AI
                </h1>
                <p style="color: #64748b; font-size: 0.85rem; margin: 0;">
                    Multi-Disease Prediction System
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Navigation Menu
        st.markdown("""
            <p style="color: #64748b; font-size: 0.75rem; font-weight: 600;
                      text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 1rem;">
                Navigation
            </p>
        """, unsafe_allow_html=True)

        # Navigation buttons with icons
        menu_items = {
            'Home': ('🏠', 'Dashboard & Overview'),
            'Diabetes': ('🩸', 'Diabetes Risk Assessment'),
            'Heart Disease': ('❤️', 'Cardiac Health Check'),
            'Parkinsons': ('🧠', "Parkinson's Screening"),
            'Liver Disease': ('🫁', 'Liver Health Analysis')
        }

        for page, (icon, description) in menu_items.items():
            is_active = st.session_state.current_page == page

            if st.button(
                f"{icon}  {page}",
                key=f"nav_{page}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.current_page = page
                st.rerun()

        st.markdown("---")

        # Information Section
        st.markdown("""
            <div style="padding: 1rem; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                        border-radius: 10px; margin-top: 1rem;">
                <p style="color: #0369a1; font-size: 0.8rem; font-weight: 600; margin-bottom: 0.5rem;">
                    ℹ️ About This Tool
                </p>
                <p style="color: #64748b; font-size: 0.75rem; line-height: 1.5;">
                    This application uses machine learning algorithms to provide health risk assessments.
                    Results are for informational purposes only.
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Footer
        st.markdown("""
            <div style="position: fixed; bottom: 1rem; padding: 0.5rem;">
                <p style="color: #94a3b8; font-size: 0.7rem; text-align: center;">
                    © 2024 HealthPredict AI<br>
                    Version 1.0.0
                </p>
            </div>
        """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="HealthPredict AI - Multi-Disease Prediction",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "# HealthPredict AI\nA professional multi-disease prediction system powered by machine learning."
        }
    )

    # Apply custom styles
    apply_custom_styles()

    # Initialize session state
    initialize_session_state()

    # Render sidebar
    render_sidebar()

    # Main content area - Route to appropriate page
    if st.session_state.current_page == 'Home':
        render_home_page()
    elif st.session_state.current_page == 'Diabetes':
        render_diabetes_page()
    elif st.session_state.current_page == 'Heart Disease':
        render_heart_page()
    elif st.session_state.current_page == 'Parkinsons':
        render_parkinsons_page()
    elif st.session_state.current_page == 'Liver Disease':
        render_liver_page()


if __name__ == "__main__":
    main()
