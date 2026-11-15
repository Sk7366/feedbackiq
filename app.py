import html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.preprocessing import preprocess_feedback
from src.sentiment import analyze_dataset
from src.insights import (
    add_categories,
    get_top_keywords,
    generate_business_insights,
)


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="FeedbackIQ — Customer Feedback Analytics",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------
# DESIGN SYSTEM TOKENS
# -----------------------------------

COLORS = {
    "bg": "#F8FAFC",
    "sidebar": "#111827",
    "sidebar_surface": "#1F2937",
    "sidebar_border": "#374151",
    "card": "#FFFFFF",
    "border": "#E2E8F0",
    "primary": "#6366F1",
    "secondary": "#14B8A6",
    "positive": "#10B981",
    "negative": "#EF4444",
    "warning": "#F59E0B",
    "neutral": "#94A3B8",
    "text": "#111827",
    "muted": "#64748B",
}

SENTIMENT_COLOR_MAP = {
    "Positive": COLORS["positive"],
    "Neutral": COLORS["neutral"],
    "Negative": COLORS["negative"],
}

NAV_SECTIONS = [
    "Overview",
    "Sentiment Analytics",
    "Category Intelligence",
    "Keyword Intelligence",
    "Business Recommendations",
    "Feedback Explorer",
]


def inject_custom_css() -> None:
    """Apply refined Linear/Stripe-inspired SaaS styling."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            -webkit-font-smoothing: antialiased;
        }

        /* Main light canvas */
        [data-testid="stAppViewContainer"] {
            background-color: #F8FAFC;
            color: #111827;
        }

        [data-testid="stHeader"] {
            background-color: rgba(248, 250, 252, 0.88);
            backdrop-filter: blur(8px);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1360px;
        }

        /* Card containers created via st.container(border=True) */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            padding: 1.15rem 1.25rem !important;
            box-shadow: 0 1px 2px rgba(17, 24, 39, 0.04);
        }

        /* Sidebar Shell (#111827) */
        [data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid #1F2937;
        }

        [data-testid="stSidebar"] * {
            color: #E5E7EB;
        }

        [data-testid="stSidebar"] hr {
            border-color: #1F2937;
            margin: 0.9rem 0;
        }

        /* Brand Identity */
        .fiq-brand {
            padding: 0.15rem 0 0.35rem 0;
        }

        .fiq-brand-row {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin-bottom: 0.5rem;
        }

        .fiq-brand-name {
            font-family: 'Manrope', 'Inter', sans-serif;
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #FFFFFF !important;
            line-height: 1.2;
        }

        .fiq-brand-tag {
            font-size: 0.68rem;
            font-weight: 600;
            color: #A5B4FC !important;
            background-color: rgba(99, 102, 241, 0.16);
            border: 1px solid rgba(99, 102, 241, 0.35);
            border-radius: 999px;
            padding: 0.1rem 0.45rem;
        }

        .fiq-brand-desc {
            font-size: 0.78rem;
            line-height: 1.48;
            color: #9CA3AF !important;
            margin: 0;
        }

        .fiq-sidebar-label {
            font-size: 0.68rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9CA3AF !important;
            margin-bottom: 0.45rem;
        }

        /* Sidebar Navigation with Monochrome Inline SVG Masks */
        [data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 0.2rem;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label {
            position: relative;
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 8px;
            padding: 0.48rem 0.75rem 0.48rem 2.25rem;
            margin: 0;
            transition: background-color 0.15s ease, border-color 0.15s ease;
            cursor: pointer;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
            display: none;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label p {
            font-size: 0.85rem;
            font-weight: 500;
            color: #D1D5DB !important;
            margin: 0;
            line-height: 1.35;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background-color: #1F2937;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover p {
            color: #FFFFFF !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background-color: #1F2937;
            border: 1px solid #374151;
            box-shadow: inset 3px 0 0 #6366F1;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
            color: #FFFFFF !important;
            font-weight: 600;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label::before {
            content: "";
            position: absolute;
            left: 0.75rem;
            top: 50%;
            transform: translateY(-50%);
            width: 15px;
            height: 15px;
            background-color: #9CA3AF;
            -webkit-mask-size: contain;
            mask-size: contain;
            -webkit-mask-repeat: no-repeat;
            mask-repeat: no-repeat;
            -webkit-mask-position: center;
            mask-position: center;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked)::before {
            background-color: #6366F1;
        }

        /* Overview grid icon */
        [data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(1)::before {
            -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='7' height='7' rx='1'/%3E%3Crect x='14' y='3' width='7' height='7' rx='1'/%3E%3Crect x='14' y='14' width='7' height='7' rx='1'/%3E%3Crect x='3' y='14' width='7' height='7' rx='1'/%3E%3C/svg%3E");
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='7' height='7' rx='1'/%3E%3Crect x='14' y='3' width='7' height='7' rx='1'/%3E%3Crect x='14' y='14' width='7' height='7' rx='1'/%3E%3Crect x='3' y='14' width='7' height='7' rx='1'/%3E%3C/svg%3E");
        }

        /* Sentiment activity icon */
        [data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(2)::before {
            -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 3v18h18'/%3E%3Cpath d='m19 9-5 5-4-4-3 3'/%3E%3C/svg%3E");
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 3v18h18'/%3E%3Cpath d='m19 9-5 5-4-4-3 3'/%3E%3C/svg%3E");
        }

        /* Category layers icon */
        [data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(3)::before {
            -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolygon points='12 2 2 7 12 12 22 7 12 2'/%3E%3Cpolyline points='2 17 12 22 22 17'/%3E%3Cpolyline points='2 12 12 17 22 12'/%3E%3C/svg%3E");
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolygon points='12 2 2 7 12 12 22 7 12 2'/%3E%3Cpolyline points='2 17 12 22 22 17'/%3E%3Cpolyline points='2 12 12 17 22 12'/%3E%3C/svg%3E");
        }

        /* Keyword hash icon */
        [data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(4)::before {
            -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='4' y1='9' x2='20' y2='9'/%3E%3Cline x1='4' y1='15' x2='20' y2='15'/%3E%3Cline x1='10' y1='3' x2='8' y2='21'/%3E%3Cline x1='16' y1='3' x2='14' y2='21'/%3E%3C/svg%3E");
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='4' y1='9' x2='20' y2='9'/%3E%3Cline x1='4' y1='15' x2='20' y2='15'/%3E%3Cline x1='10' y1='3' x2='8' y2='21'/%3E%3Cline x1='16' y1='3' x2='14' y2='21'/%3E%3C/svg%3E");
        }

        /* Recommendations check-circle icon */
        [data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(5)::before {
            -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M9 11l3 3L22 4'/%3E%3Cpath d='M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11'/%3E%3C/svg%3E");
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M9 11l3 3L22 4'/%3E%3Cpath d='M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11'/%3E%3C/svg%3E");
        }

        /* Explorer search icon */
        [data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(6)::before {
            -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'/%3E%3C/svg%3E");
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'/%3E%3C/svg%3E");
        }

        /* Sidebar file uploader contrast */
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background-color: #1F2937;
            border: 1px dashed #374151;
            border-radius: 10px;
            padding: 0.75rem;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
            color: #9CA3AF !important;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
            background-color: #6366F1;
            color: #FFFFFF !important;
            border: none;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 500;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover {
            background-color: #4F46E5;
        }

        /* Dataset status card in sidebar */
        .fiq-dataset-status {
            background-color: #1F2937;
            border: 1px solid #374151;
            border-radius: 10px;
            padding: 0.7rem 0.85rem;
            margin-top: 0.6rem;
        }

        .fiq-status-row {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.78rem;
            font-weight: 600;
            color: #F9FAFB !important;
            margin-bottom: 0.35rem;
        }

        .fiq-status-meta {
            font-size: 0.74rem;
            color: #9CA3AF !important;
            line-height: 1.5;
        }

        .fiq-dot-live {
            width: 7px;
            height: 7px;
            border-radius: 999px;
            background-color: #10B981;
            display: inline-block;
            flex-shrink: 0;
        }

        /* Page Header Typography */
        .fiq-header {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: flex-end;
            gap: 1rem;
            padding-bottom: 1rem;
            margin-bottom: 1.25rem;
            border-bottom: 1px solid #E2E8F0;
        }

        .fiq-kicker {
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #6366F1;
            margin-bottom: 0.2rem;
        }

        .fiq-title {
            font-family: 'Manrope', 'Inter', sans-serif;
            font-size: 1.55rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            color: #111827;
            margin: 0;
            line-height: 1.25;
        }

        .fiq-subtitle {
            font-size: 0.875rem;
            font-weight: 400;
            color: #64748B;
            margin: 0.25rem 0 0 0;
            line-height: 1.45;
        }

        .fiq-header-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            font-size: 0.76rem;
            font-weight: 500;
            color: #334155;
            box-shadow: 0 1px 2px rgba(17, 24, 39, 0.03);
        }

        /* Section & Card Headings */
        .fiq-card-head {
            margin-bottom: 0.75rem;
        }

        .fiq-card-title {
            font-family: 'Manrope', 'Inter', sans-serif;
            font-size: 0.98rem;
            font-weight: 600;
            letter-spacing: -0.01em;
            color: #111827;
            margin: 0;
            line-height: 1.35;
        }

        .fiq-card-desc {
            font-size: 0.8rem;
            color: #64748B;
            margin: 0.15rem 0 0 0;
            line-height: 1.4;
        }

        /* KPI Cards */
        .fiq-kpi {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.05rem 1.15rem;
            box-shadow: 0 1px 2px rgba(17, 24, 39, 0.04);
            height: 100%;
        }

        .fiq-kpi-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.6rem;
        }

        .fiq-kpi-label {
            font-size: 0.78rem;
            font-weight: 500;
            color: #64748B;
            line-height: 1.3;
        }

        .fiq-kpi-icon {
            width: 28px;
            height: 28px;
            border-radius: 8px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        .fiq-kpi-value {
            font-family: 'Manrope', 'Inter', sans-serif;
            font-size: 1.55rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            color: #111827;
            line-height: 1.15;
            font-variant-numeric: tabular-nums;
        }

        .fiq-kpi-foot {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            margin-top: 0.45rem;
            font-size: 0.76rem;
            color: #64748B;
            line-height: 1.35;
        }

        .fiq-kpi-indicator {
            width: 6px;
            height: 6px;
            border-radius: 999px;
            display: inline-block;
            flex-shrink: 0;
        }

        /* Recent Feedback Activity Rows */
        .fiq-activity-item {
            padding: 0.72rem 0.85rem;
            border: 1px solid #F1F5F9;
            background-color: #F8FAFC;
            border-radius: 9px;
            margin-bottom: 0.55rem;
        }

        .fiq-activity-item:last-child {
            margin-bottom: 0;
        }

        .fiq-activity-text {
            font-size: 0.83rem;
            color: #1F2937;
            line-height: 1.48;
            margin: 0 0 0.45rem 0;
        }

        .fiq-activity-meta {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.72rem;
            color: #64748B;
        }

        /* Semantic Pills & Category Tags */
        .fiq-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.14rem 0.52rem;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 600;
            line-height: 1.35;
        }

        .fiq-pill-positive {
            background-color: #ECFDF5;
            color: #065F46;
            border: 1px solid #A7F3D0;
        }

        .fiq-pill-negative {
            background-color: #FEF2F2;
            color: #991B1B;
            border: 1px solid #FECACA;
        }

        .fiq-pill-neutral {
            background-color: #F1F5F9;
            color: #475569;
            border: 1px solid #CBD5E1;
        }

        .fiq-pill-high {
            background-color: #FEF2F2;
            color: #991B1B;
            border: 1px solid #FECACA;
        }

        .fiq-pill-medium {
            background-color: #FFFBEB;
            color: #92400E;
            border: 1px solid #FDE68A;
        }

        .fiq-pill-low {
            background-color: #ECFDF5;
            color: #065F46;
            border: 1px solid #A7F3D0;
        }

        .fiq-tag {
            background-color: #EEF2FF;
            color: #4338CA;
            border-radius: 6px;
            padding: 0.12rem 0.45rem;
            font-size: 0.7rem;
            font-weight: 500;
        }

        /* Accessible Callout & Recommendation Cards */
        .fiq-callout {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #6366F1;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            box-shadow: 0 1px 2px rgba(17, 24, 39, 0.04);
            margin-bottom: 1rem;
        }

        .fiq-callout-label {
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #4F46E5;
            margin-bottom: 0.3rem;
        }

        .fiq-callout-text {
            font-size: 0.94rem;
            font-weight: 600;
            color: #111827;
            line-height: 1.48;
            margin: 0;
        }

        .fiq-rec-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.05rem 1.2rem;
            box-shadow: 0 1px 2px rgba(17, 24, 39, 0.04);
            margin-bottom: 0.85rem;
        }

        .fiq-rec-card-urgent {
            border-left: 4px solid #EF4444;
            background-color: #FFFAFA;
        }

        .fiq-rec-card-medium {
            border-left: 4px solid #F59E0B;
        }

        .fiq-rec-card-low {
            border-left: 4px solid #10B981;
        }

        .fiq-rec-top {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.45rem;
        }

        .fiq-rec-title {
            font-family: 'Manrope', 'Inter', sans-serif;
            font-size: 0.96rem;
            font-weight: 600;
            color: #111827;
            margin: 0;
        }

        .fiq-rec-stats {
            display: flex;
            flex-wrap: wrap;
            gap: 1.25rem;
            font-size: 0.78rem;
            color: #475569;
            margin-bottom: 0.55rem;
        }

        .fiq-rec-action {
            font-size: 0.86rem;
            color: #1F2937;
            line-height: 1.5;
            margin: 0;
            padding-top: 0.55rem;
            border-top: 1px solid #F1F5F9;
        }

        /* Empty / Informational States (High Contrast) */
        .fiq-empty {
            background-color: #F8FAFC;
            border: 1px dashed #CBD5E1;
            border-radius: 10px;
            padding: 1.5rem 1.25rem;
            text-align: center;
            font-size: 0.84rem;
            color: #475569;
            line-height: 1.5;
        }

        /* Export button polish */
        .stDownloadButton button {
            background-color: #6366F1;
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            font-size: 0.84rem;
            font-weight: 600;
            padding: 0.52rem 1rem;
        }

        .stDownloadButton button:hover {
            background-color: #4F46E5;
            color: #FFFFFF;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------
# UI HELPER FUNCTIONS
# -----------------------------------

KPI_ICONS = {
    "volume": """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>""",
    "positive": """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>""",
    "negative": """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>""",
    "rating": """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>""",
    "neutral": """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/></svg>""",
    "category": """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>""",
    "keyword": """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1='4' y1='9' x2='20' y2='9'/><line x1='4' y1='15' x2='20' y2='15'/><line x1='10' y1='3' x2='8' y2='21'/><line x1='16' y1='3' x2='14' y2='21'/></svg>""",
}


def render_kpi_card(
    label: str,
    value: str,
    subtitle: str,
    accent: str,
    tint: str,
    icon_key: str = "volume",
) -> None:
    """Render a compact, high-contrast SaaS KPI card."""
    icon_svg = KPI_ICONS.get(icon_key, KPI_ICONS["volume"])
    st.markdown(
        f"""
        <div class="fiq-kpi">
            <div class="fiq-kpi-header">
                <span class="fiq-kpi-label">{html.escape(label)}</span>
                <span class="fiq-kpi-icon" style="background-color: {tint}; color: {accent};">
                    {icon_svg}
                </span>
            </div>
            <div class="fiq-kpi-value">{html.escape(str(value))}</div>
            <div class="fiq-kpi-foot">
                <span class="fiq-kpi-indicator" style="background-color: {accent};"></span>
                <span>{html.escape(subtitle)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(
    kicker: str,
    title: str,
    subtitle: str,
    source_label: str,
    total_records: int,
) -> None:
    """Render consistent top page header with dataset context pill."""
    st.markdown(
        f"""
        <div class="fiq-header">
            <div>
                <div class="fiq-kicker">{html.escape(kicker)}</div>
                <h1 class="fiq-title">{html.escape(title)}</h1>
                <p class="fiq-subtitle">{html.escape(subtitle)}</p>
            </div>
            <div class="fiq-header-pill">
                <span class="fiq-dot-live"></span>
                <span>{html.escape(source_label)} &middot; {total_records:,} feedback entries</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_card_heading(title: str, description: str) -> None:
    """Render card title and subtitle inside st.container(border=True)."""
    st.markdown(
        f"""
        <div class="fiq-card-head">
            <h3 class="fiq-card-title">{html.escape(title)}</h3>
            <p class="fiq-card-desc">{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_theme(fig: go.Figure, height: int = 330) -> go.Figure:
    """Apply clean Stripe/Linear-inspired styling to Plotly figures."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            color="#334155",
            size=12,
        ),
        margin=dict(l=8, r=18, t=24, b=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text="",
            font=dict(size=11, color="#475569"),
        ),
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="#1F2937",
            font=dict(color="#F9FAFB", size=12),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#E2E8F0",
        tickfont=dict(color="#64748B", size=11),
        title_font=dict(color="#475569", size=11),
    )
    fig.update_yaxes(
        gridcolor="#F1F5F9",
        zerolinecolor="#E2E8F0",
        tickfont=dict(color="#64748B", size=11),
        title_font=dict(color="#475569", size=11),
    )
    return fig


def get_valid_rating_df(df: pd.DataFrame) -> pd.DataFrame:
    """Safely extract rows with valid numeric ratings."""
    if "rating" not in df.columns:
        return pd.DataFrame()
    rating_df = df.copy()
    rating_df["rating"] = pd.to_numeric(rating_df["rating"], errors="coerce")
    return rating_df.dropna(subset=["rating"])


def get_valid_date_df(df: pd.DataFrame) -> pd.DataFrame:
    """Safely extract rows with valid parsed dates."""
    if "date" not in df.columns:
        return pd.DataFrame()
    date_df = df.copy()
    date_df["date"] = pd.to_datetime(date_df["date"], errors="coerce")
    return date_df.dropna(subset=["date"])


def build_sentiment_df(insights: dict) -> pd.DataFrame:
    """Create ordered sentiment summary covering Positive, Neutral, and Negative."""
    counts = insights["sentiment_counts"]
    percentages = insights["sentiment_percentages"]
    return pd.DataFrame(
        [
            {
                "Sentiment": label,
                "Count": int(counts.get(label, 0)),
                "Percentage": percentages.get(label, 0.0),
            }
            for label in ["Positive", "Neutral", "Negative"]
        ]
    )


# -----------------------------------
# CACHED NLP PIPELINE
# -----------------------------------


@st.cache_data(show_spinner=False)
def run_nlp_pipeline(raw_df: pd.DataFrame):
    """Execute preprocessing, VADER sentiment, categorization, and insights."""
    # Drop pre-computed columns if user re-uploads an exported FeedbackIQ CSV
    computed_cols = [
        "clean_feedback",
        "positive_score",
        "negative_score",
        "neutral_score",
        "compound_score",
        "sentiment",
        "categories",
    ]
    cleaned_input = raw_df.drop(
        columns=[col for col in computed_cols if col in raw_df.columns]
    )

    processed_df = preprocess_feedback(cleaned_input)
    if processed_df.empty:
        return processed_df, None

    analysed_df = analyze_dataset(processed_df)
    analysed_df = add_categories(analysed_df)
    insights = generate_business_insights(analysed_df)
    return analysed_df, insights


# -----------------------------------
# INITIALIZE STYLES
# -----------------------------------

inject_custom_css()


# -----------------------------------
# SIDEBAR (BRANDING, NAV, DATA SOURCE)
# -----------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="fiq-brand">
            <div class="fiq-brand-row">
                <svg width="30" height="30" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect width="32" height="32" rx="8" fill="#6366F1"/>
                    <path d="M9 21V15M16 21V10M23 21V13" stroke="#FFFFFF" stroke-width="2.4" stroke-linecap="round"/>
                    <circle cx="16" cy="10" r="2" fill="#14B8A6"/>
                </svg>
                <span class="fiq-brand-name">FeedbackIQ</span>
                <span class="fiq-brand-tag">NLP</span>
            </div>
            <p class="fiq-brand-desc">
                Customer feedback intelligence for sentiment tracking, category risk detection, and prioritized actions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown('<div class="fiq-sidebar-label">Analytics</div>', unsafe_allow_html=True)
    active_section = st.radio(
        "Navigation",
        NAV_SECTIONS,
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown('<div class="fiq-sidebar-label">Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload customer feedback CSV",
        type=["csv"],
        help="Required column: 'feedback'. Optional columns: 'date' and 'rating'.",
    )

    use_sample = st.checkbox(
        "Use sample feedback dataset",
        value=uploaded_file is None,
    )


# -----------------------------------
# DATA LOADING & VALIDATION
# -----------------------------------

raw_df = None
source_label = ""

try:
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        source_label = uploaded_file.name
    elif use_sample:
        raw_df = pd.read_csv("data/sample_feedback.csv")
        source_label = "Sample Dataset"
    else:
        with st.container(border=True):
            st.markdown(
                """
                <div style="text-align:center; padding: 2rem 1rem;">
                    <div class="fiq-kicker">FeedbackIQ Workspace</div>
                    <h2 class="fiq-title" style="margin-bottom: 0.4rem;">Select a dataset to begin</h2>
                    <p class="fiq-subtitle" style="max-width: 500px; margin: 0 auto;">
                        Upload a CSV containing a <code>feedback</code> column from the sidebar,
                        or enable <strong>Use sample feedback dataset</strong> to inspect live analytics.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.stop()

    if "feedback" not in raw_df.columns:
        st.error("Your CSV must contain a column named 'feedback'.")
        st.stop()

    with st.spinner("Analysing customer feedback..."):
        df, insights = run_nlp_pipeline(raw_df)

    if df.empty or insights is None:
        st.error("No valid customer feedback found.")
        st.stop()

except FileNotFoundError:
    st.error(
        "Sample dataset not found at 'data/sample_feedback.csv'. "
        "Please upload a CSV file from the sidebar."
    )
    st.stop()
except Exception as exc:
    st.error(f"Unable to process the dataset: {exc}")
    st.stop()


rating_df_global = get_valid_rating_df(df)
date_df_global = get_valid_date_df(df)

with st.sidebar:
    detected_fields = ["Text"]
    if not date_df_global.empty:
        detected_fields.append("Date")
    if not rating_df_global.empty:
        detected_fields.append("Rating")

    st.markdown(
        f"""
        <div class="fiq-dataset-status">
            <div class="fiq-status-row">
                <span class="fiq-dot-live"></span>
                <span>{html.escape(source_label)}</span>
            </div>
            <div class="fiq-status-meta">
                Records analysed: <strong>{insights['total_feedback']:,}</strong><br/>
                Fields detected: {html.escape(', '.join(detected_fields))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------
# SECTION 1: OVERVIEW
# -----------------------------------


def render_overview() -> None:
    render_page_header(
        kicker="Overview",
        title="Customer Feedback Overview",
        subtitle="Key satisfaction metrics, sentiment mix, and latest customer activity.",
        source_label=source_label,
        total_records=insights["total_feedback"],
    )

    total = insights["total_feedback"]
    counts = insights["sentiment_counts"]
    pct = insights["sentiment_percentages"]

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        render_kpi_card(
            label="Total Feedback Analysed",
            value=f"{total:,}",
            subtitle=f"{len(insights['category_counts'])} categories identified",
            accent=COLORS["primary"],
            tint="#EEF2FF",
            icon_key="volume",
        )
    with c2:
        render_kpi_card(
            label="Positive Sentiment",
            value=f"{pct['Positive']}%",
            subtitle=f"{counts.get('Positive', 0):,} positive entries",
            accent=COLORS["positive"],
            tint="#ECFDF5",
            icon_key="positive",
        )
    with c3:
        render_kpi_card(
            label="Negative Sentiment",
            value=f"{pct['Negative']}%",
            subtitle=f"{counts.get('Negative', 0):,} negative entries",
            accent=COLORS["negative"],
            tint="#FEF2F2",
            icon_key="negative",
        )
    with c4:
        if not rating_df_global.empty:
            avg_rating = rating_df_global["rating"].mean()
            render_kpi_card(
                label="Average Customer Rating",
                value=f"{avg_rating:.2f} / 5",
                subtitle=f"Across {len(rating_df_global):,} rated reviews",
                accent=COLORS["warning"],
                tint="#FFFBEB",
                icon_key="rating",
            )
        else:
            render_kpi_card(
                label="Neutral Sentiment",
                value=f"{pct['Neutral']}%",
                subtitle=f"{counts.get('Neutral', 0):,} neutral entries",
                accent=COLORS["neutral"],
                tint="#F1F5F9",
                icon_key="neutral",
            )

    st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.0, 1.3], gap="medium")

    with col_left:
        with st.container(border=True):
            render_card_heading(
                "Sentiment Distribution",
                "Share of feedback classified as Positive, Neutral, or Negative",
            )
            sentiment_df = build_sentiment_df(insights)
            fig = px.pie(
                sentiment_df,
                names="Sentiment",
                values="Count",
                hole=0.64,
                color="Sentiment",
                color_discrete_map=SENTIMENT_COLOR_MAP,
            )
            fig.update_traces(
                textposition="outside",
                textinfo="percent+label",
                textfont=dict(size=11, color="#334155"),
                marker=dict(line=dict(color="#FFFFFF", width=2.5)),
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
            )
            fig.add_annotation(
                text=f"<b>{total:,}</b><br><span style='font-size:11px;color:#64748B;'>Total</span>",
                x=0.5,
                y=0.5,
                font=dict(size=19, color="#111827", family="Manrope, Inter, sans-serif"),
                showarrow=False,
            )
            apply_plotly_theme(fig, height=340)
            fig.update_layout(showlegend=False, margin=dict(l=16, r=16, t=20, b=20))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with col_right:
        with st.container(border=True):
            render_card_heading(
                "Recent Feedback Activity",
                "Latest customer comments with VADER sentiment and assigned topics",
            )

            if not date_df_global.empty:
                recent_df = date_df_global.sort_values("date", ascending=False).head(5)
            else:
                recent_df = df.tail(5).iloc[::-1]

            activity_blocks = []
            for _, row in recent_df.iterrows():
                text_val = str(row["feedback"])
                snippet = (
                    text_val
                    if len(text_val) <= 155
                    else text_val[:152].rstrip() + "..."
                )
                sentiment = str(row["sentiment"])
                pill_cls = f"fiq-pill-{sentiment.lower()}"
                cats = row.get("categories", ["Other"])
                cat_html = "".join(
                    f'<span class="fiq-tag">{html.escape(str(c))}</span>'
                    for c in cats
                )

                meta_bits = []
                if "date" in row and pd.notna(row["date"]):
                    d_str = (
                        row["date"].strftime("%Y-%m-%d")
                        if hasattr(row["date"], "strftime")
                        else str(row["date"])
                    )
                    meta_bits.append(html.escape(d_str))
                if "rating" in row and pd.notna(row["rating"]):
                    try:
                        meta_bits.append(f"Rating {float(row['rating']):.1f}/5")
                    except (TypeError, ValueError):
                        pass

                meta_suffix = (
                    f"<span>{' &middot; '.join(meta_bits)}</span>" if meta_bits else ""
                )

                activity_blocks.append(
                    f"""
                    <div class="fiq-activity-item">
                        <p class="fiq-activity-text">{html.escape(snippet)}</p>
                        <div class="fiq-activity-meta">
                            <span class="fiq-pill {pill_cls}">{html.escape(sentiment)}</span>
                            {cat_html}
                            {meta_suffix}
                        </div>
                    </div>
                    """
                )

            st.markdown("".join(activity_blocks), unsafe_allow_html=True)

    st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3, gap="medium")
    with s1:
        most_discussed = insights["most_discussed_category"]
        render_kpi_card(
            label="Most Discussed Category",
            value=most_discussed,
            subtitle=f"{insights['category_counts'].get(most_discussed, 0):,} customer mentions",
            accent=COLORS["primary"],
            tint="#EEF2FF",
            icon_key="category",
        )
    with s2:
        concerning = insights["most_concerning_category"]
        concerning_neg = insights["negative_category_counts"].get(concerning, 0)
        render_kpi_card(
            label="Most Concerning Category",
            value=concerning,
            subtitle=(
                f"{concerning_neg:,} negative reviews"
                if concerning != "None"
                else "No category meets minimum threshold"
            ),
            accent=COLORS["negative"],
            tint="#FEF2F2",
            icon_key="negative",
        )
    with s3:
        top_kw = insights["top_keywords"][0][0] if insights["top_keywords"] else "N/A"
        top_kw_cnt = insights["top_keywords"][0][1] if insights["top_keywords"] else 0
        render_kpi_card(
            label="Top Recurring Keyword",
            value=top_kw.title() if top_kw != "N/A" else "N/A",
            subtitle=f"Appears {top_kw_cnt:,} times across feedback",
            accent=COLORS["secondary"],
            tint="#F0FDFA",
            icon_key="keyword",
        )


# -----------------------------------
# SECTION 2: SENTIMENT ANALYTICS
# -----------------------------------


def render_sentiment_analytics() -> None:
    render_page_header(
        kicker="Sentiment Analytics",
        title="Polarity, Trends & Rating Alignment",
        subtitle="Track sentiment distribution, daily shifts over time, and alignment with customer ratings.",
        source_label=source_label,
        total_records=insights["total_feedback"],
    )

    counts = insights["sentiment_counts"]
    pct = insights["sentiment_percentages"]
    mean_compound = float(df["compound_score"].mean())

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        render_kpi_card(
            label="Positive Feedback",
            value=f"{counts.get('Positive', 0):,}",
            subtitle=f"{pct['Positive']}% of total",
            accent=COLORS["positive"],
            tint="#ECFDF5",
            icon_key="positive",
        )
    with k2:
        render_kpi_card(
            label="Neutral Feedback",
            value=f"{counts.get('Neutral', 0):,}",
            subtitle=f"{pct['Neutral']}% of total",
            accent=COLORS["neutral"],
            tint="#F1F5F9",
            icon_key="neutral",
        )
    with k3:
        render_kpi_card(
            label="Negative Feedback",
            value=f"{counts.get('Negative', 0):,}",
            subtitle=f"{pct['Negative']}% of total",
            accent=COLORS["negative"],
            tint="#FEF2F2",
            icon_key="negative",
        )
    with k4:
        render_kpi_card(
            label="Mean Compound Score",
            value=f"{mean_compound:+.3f}",
            subtitle="VADER polarity (-1.0 to +1.0)",
            accent=COLORS["primary"],
            tint="#EEF2FF",
            icon_key="volume",
        )

    st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)

    sentiment_df = build_sentiment_df(insights)
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        with st.container(border=True):
            render_card_heading(
                "Sentiment Distribution",
                "Proportional share across sentiment classes",
            )
            fig_donut = px.pie(
                sentiment_df,
                names="Sentiment",
                values="Count",
                hole=0.6,
                color="Sentiment",
                color_discrete_map=SENTIMENT_COLOR_MAP,
            )
            fig_donut.update_traces(
                textinfo="percent+label",
                textfont=dict(size=11, color="#334155"),
                marker=dict(line=dict(color="#FFFFFF", width=2.5)),
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
            )
            apply_plotly_theme(fig_donut, height=320)
            fig_donut.update_layout(showlegend=False)
            st.plotly_chart(fig_donut, width="stretch", config={"displayModeBar": False})

    with c2:
        with st.container(border=True):
            render_card_heading(
                "Sentiment Volume",
                "Total feedback count per predicted sentiment",
            )
            fig_vol = px.bar(
                sentiment_df,
                x="Sentiment",
                y="Count",
                color="Sentiment",
                color_discrete_map=SENTIMENT_COLOR_MAP,
                text="Count",
            )
            fig_vol.update_traces(
                textposition="outside",
                cliponaxis=False,
                marker_line_width=0,
                hovertemplate="<b>%{x}</b><br>Reviews: %{y:,}<extra></extra>",
            )
            apply_plotly_theme(fig_vol, height=320)
            fig_vol.update_layout(showlegend=False, xaxis_title="", yaxis_title="Reviews")
            st.plotly_chart(fig_vol, width="stretch", config={"displayModeBar": False})

    st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        render_card_heading(
            "Sentiment Trends Over Time",
            "Daily customer sentiment volume across the feedback timeline",
        )
        if not date_df_global.empty:
            trend_df = date_df_global.copy()
            trend_df["date"] = trend_df["date"].dt.date
            daily_sentiment = (
                trend_df.groupby(["date", "sentiment"])
                .size()
                .reset_index(name="Feedback Count")
            )
            fig_trend = px.line(
                daily_sentiment,
                x="date",
                y="Feedback Count",
                color="sentiment",
                markers=True,
                color_discrete_map=SENTIMENT_COLOR_MAP,
                category_orders={"sentiment": ["Positive", "Neutral", "Negative"]},
            )
            fig_trend.update_traces(
                line=dict(width=2.25),
                marker=dict(size=6),
                hovertemplate="%{fullData.name}: %{y:,}<extra></extra>",
            )
            apply_plotly_theme(fig_trend, height=330)
            fig_trend.update_layout(
                hovermode="x unified",
                xaxis_title="Date",
                yaxis_title="Feedback Count",
            )
            st.plotly_chart(fig_trend, width="stretch", config={"displayModeBar": False})
        else:
            st.markdown(
                """
                <div class="fiq-empty">
                    No valid <code>date</code> column found in the active dataset.
                    Add a <code>date</code> column (e.g. <code>YYYY-MM-DD</code>) to view time-based sentiment trends.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        render_card_heading(
            "Customer Rating vs. Sentiment",
            "Comparison between numeric star ratings and VADER sentiment classifications",
        )
        if not rating_df_global.empty:
            r1, r2 = st.columns(2, gap="medium")
            with r1:
                rating_counts = (
                    rating_df_global["rating"]
                    .value_counts()
                    .sort_index()
                    .reset_index()
                )
                rating_counts.columns = ["Rating", "Count"]
                fig_rating = px.bar(
                    rating_counts,
                    x="Rating",
                    y="Count",
                    text="Count",
                    color_discrete_sequence=[COLORS["primary"]],
                )
                fig_rating.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="Rating %{x}: %{y:,} reviews<extra></extra>",
                )
                apply_plotly_theme(fig_rating, height=300)
                fig_rating.update_layout(
                    title=dict(
                        text="Rating Distribution",
                        font=dict(size=13, color="#111827"),
                        x=0,
                    ),
                    xaxis_title="Rating",
                    yaxis_title="Count",
                )
                st.plotly_chart(fig_rating, width="stretch", config={"displayModeBar": False})

            with r2:
                sentiment_rating = (
                    rating_df_global.groupby("sentiment")["rating"]
                    .mean()
                    .reindex(["Positive", "Neutral", "Negative"])
                    .dropna()
                    .reset_index()
                )
                fig_sr = px.bar(
                    sentiment_rating,
                    x="sentiment",
                    y="rating",
                    color="sentiment",
                    color_discrete_map=SENTIMENT_COLOR_MAP,
                    text_auto=".2f",
                )
                fig_sr.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="<b>%{x}</b><br>Avg Rating: %{y:.2f} / 5<extra></extra>",
                )
                apply_plotly_theme(fig_sr, height=300)
                fig_sr.update_layout(
                    showlegend=False,
                    title=dict(
                        text="Average Rating by Predicted Sentiment",
                        font=dict(size=13, color="#111827"),
                        x=0,
                    ),
                    xaxis_title="",
                    yaxis_title="Average Rating",
                    yaxis=dict(range=[0, 5.7]),
                )
                st.plotly_chart(fig_sr, width="stretch", config={"displayModeBar": False})
        else:
            st.markdown(
                """
                <div class="fiq-empty">
                    No valid <code>rating</code> column found in the active dataset.
                    Add numeric ratings (e.g. 1–5) to enable rating vs. sentiment analytics.
                </div>
                """,
                unsafe_allow_html=True,
            )


# -----------------------------------
# SECTION 3: CATEGORY INTELLIGENCE
# -----------------------------------


def render_category_intelligence() -> None:
    render_page_header(
        kicker="Category Intelligence",
        title="Business Area & Risk Breakdown",
        subtitle="Identify high-volume themes, negative feedback rates, and operational priority tiers.",
        source_label=source_label,
        total_records=insights["total_feedback"],
    )

    category_analysis = insights["category_analysis"]
    priority_df = pd.DataFrame(
        [
            {
                "Category": category,
                "Total Reviews": metrics["total_reviews"],
                "Negative Reviews": metrics["negative_reviews"],
                "Negative Rate (%)": metrics["negative_percentage"],
                "Priority": metrics["priority"],
            }
            for category, metrics in category_analysis.items()
        ]
    )

    high_cnt = int((priority_df["Priority"] == "High").sum()) if not priority_df.empty else 0
    med_cnt = int((priority_df["Priority"] == "Medium").sum()) if not priority_df.empty else 0
    low_cnt = int((priority_df["Priority"] == "Low").sum()) if not priority_df.empty else 0

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        most_discussed = insights["most_discussed_category"]
        render_kpi_card(
            label="Most Discussed Category",
            value=most_discussed,
            subtitle=f"{insights['category_counts'].get(most_discussed, 0):,} mentions",
            accent=COLORS["primary"],
            tint="#EEF2FF",
            icon_key="category",
        )
    with k2:
        concerning = insights["most_concerning_category"]
        concerning_rate = insights["category_negative_rates"].get(concerning, 0)
        render_kpi_card(
            label="Most Concerning Category",
            value=concerning,
            subtitle=(
                f"{concerning_rate}% negative rate"
                if concerning != "None"
                else "Insufficient category volume"
            ),
            accent=COLORS["negative"],
            tint="#FEF2F2",
            icon_key="negative",
        )
    with k3:
        render_kpi_card(
            label="High & Medium Priority",
            value=f"{high_cnt} High / {med_cnt} Med",
            subtitle="Categories needing review",
            accent=COLORS["warning"],
            tint="#FFFBEB",
            icon_key="negative",
        )
    with k4:
        render_kpi_card(
            label="Low Priority Categories",
            value=str(low_cnt),
            subtitle="Within acceptable thresholds",
            accent=COLORS["positive"],
            tint="#ECFDF5",
            icon_key="positive",
        )

    st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        with st.container(border=True):
            render_card_heading(
                "Category Distribution",
                "Total feedback mentions per business category",
            )
            cat_df = pd.DataFrame(
                list(insights["category_counts"].items()),
                columns=["Category", "Feedback Count"],
            )
            if not cat_df.empty:
                fig_cat = px.bar(
                    cat_df.sort_values("Feedback Count", ascending=True),
                    x="Feedback Count",
                    y="Category",
                    orientation="h",
                    text="Feedback Count",
                    color_discrete_sequence=[COLORS["primary"]],
                )
                fig_cat.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="<b>%{y}</b><br>Mentions: %{x:,}<extra></extra>",
                )
                apply_plotly_theme(fig_cat, height=340)
                fig_cat.update_layout(xaxis_title="Feedback Count", yaxis_title="")
                st.plotly_chart(fig_cat, width="stretch", config={"displayModeBar": False})
            else:
                st.markdown('<div class="fiq-empty">No category data available.</div>', unsafe_allow_html=True)

    with c2:
        with st.container(border=True):
            render_card_heading(
                "Category-wise Negative Rate",
                "Share of negative reviews within each category",
            )
            neg_df = pd.DataFrame(
                list(insights["category_negative_rates"].items()),
                columns=["Category", "Negative Rate (%)"],
            )
            if not neg_df.empty:
                fig_neg = px.bar(
                    neg_df.sort_values("Negative Rate (%)", ascending=True),
                    x="Negative Rate (%)",
                    y="Category",
                    orientation="h",
                    text_auto=".1f",
                    color="Negative Rate (%)",
                    color_continuous_scale=[COLORS["secondary"], COLORS["warning"], COLORS["negative"]],
                )
                fig_neg.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="<b>%{y}</b><br>Negative Rate: %{x:.2f}%<extra></extra>",
                )
                apply_plotly_theme(fig_neg, height=340)
                fig_neg.update_layout(
                    coloraxis_showscale=False,
                    xaxis_title="Negative Rate (%)",
                    yaxis_title="",
                )
                st.plotly_chart(fig_neg, width="stretch", config={"displayModeBar": False})
            else:
                st.markdown('<div class="fiq-empty">No negative rate data available.</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        render_card_heading(
            "Category Priority Matrix",
            "Sorted by negative review volume and percentage to guide team triage",
        )
        if not priority_df.empty:
            priority_df = priority_df.sort_values(
                by=["Negative Reviews", "Negative Rate (%)"],
                ascending=False,
            )
            st.dataframe(
                priority_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Category": st.column_config.TextColumn("Category", width="medium"),
                    "Total Reviews": st.column_config.NumberColumn("Total Reviews", format="%d"),
                    "Negative Reviews": st.column_config.NumberColumn("Negative Reviews", format="%d"),
                    "Negative Rate (%)": st.column_config.ProgressColumn(
                        "Negative Rate",
                        min_value=0,
                        max_value=100,
                        format="%.2f%%",
                    ),
                    "Priority": st.column_config.TextColumn("Priority", width="small"),
                },
            )
        else:
            st.markdown('<div class="fiq-empty">No category priority metrics available.</div>', unsafe_allow_html=True)


# -----------------------------------
# SECTION 4: KEYWORD INTELLIGENCE
# -----------------------------------


def render_keyword_intelligence() -> None:
    render_page_header(
        kicker="Keyword Intelligence",
        title="Dominant Terms & Keyword Density",
        subtitle="Surface the most frequent meaningful terms mentioned across customer feedback.",
        source_label=source_label,
        total_records=insights["total_feedback"],
    )

    top_n = st.slider(
        "Number of keywords to display",
        min_value=5,
        max_value=25,
        value=10,
        step=5,
    )

    keywords = get_top_keywords(df, top_n=top_n)
    keyword_df = pd.DataFrame(keywords, columns=["Keyword", "Frequency"])

    if keyword_df.empty:
        with st.container(border=True):
            st.markdown(
                '<div class="fiq-empty">No meaningful keywords extracted from this dataset.</div>',
                unsafe_allow_html=True,
            )
        return

    total_mentions = int(keyword_df["Frequency"].sum())
    keyword_df["Share (%)"] = (keyword_df["Frequency"] / total_mentions * 100).round(1)

    c1, c2 = st.columns([1.15, 1.0], gap="medium")

    with c1:
        with st.container(border=True):
            render_card_heading(
                "Keyword Frequency Ranking",
                f"Top {len(keyword_df)} recurring words after stopword filtering",
            )
            fig_kw = px.bar(
                keyword_df.sort_values("Frequency", ascending=True),
                x="Frequency",
                y="Keyword",
                orientation="h",
                text="Frequency",
                color="Frequency",
                color_continuous_scale=["#99F6E4", COLORS["secondary"], COLORS["primary"]],
            )
            fig_kw.update_traces(
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>Mentions: %{x:,}<extra></extra>",
            )
            apply_plotly_theme(fig_kw, height=max(340, len(keyword_df) * 28))
            fig_kw.update_layout(
                coloraxis_showscale=False,
                xaxis_title="Occurrences",
                yaxis_title="",
            )
            st.plotly_chart(fig_kw, width="stretch", config={"displayModeBar": False})

    with c2:
        with st.container(border=True):
            render_card_heading(
                "Keyword Density Map",
                "Relative concentration of top extracted customer terms",
            )
            fig_tree = px.treemap(
                keyword_df,
                path=["Keyword"],
                values="Frequency",
                color="Frequency",
                color_continuous_scale=["#CCFBF1", COLORS["secondary"], COLORS["primary"]],
            )
            fig_tree.update_traces(
                textinfo="label+value",
                textfont=dict(size=12),
                hovertemplate="<b>%{label}</b><br>Mentions: %{value:,}<extra></extra>",
            )
            apply_plotly_theme(fig_tree, height=280)
            fig_tree.update_layout(
                coloraxis_showscale=False,
                margin=dict(l=4, r=4, t=8, b=4),
            )
            st.plotly_chart(fig_tree, width="stretch", config={"displayModeBar": False})

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            render_card_heading(
                "Keyword Breakdown",
                "Exact frequency and share across displayed top terms",
            )
            st.dataframe(
                keyword_df,
                width="stretch",
                hide_index=True,
                height=240,
                column_config={
                    "Share (%)": st.column_config.ProgressColumn(
                        "Relative Share",
                        min_value=0,
                        max_value=100,
                        format="%.1f%%",
                    )
                },
            )


# -----------------------------------
# SECTION 5: BUSINESS RECOMMENDATIONS
# -----------------------------------


def render_business_recommendations() -> None:
    render_page_header(
        kicker="Business Recommendations",
        title="Prioritized Action Plan",
        subtitle="Category-specific operational recommendations ranked by urgency and negative impact.",
        source_label=source_label,
        total_records=insights["total_feedback"],
    )

    st.markdown(
        f"""
        <div class="fiq-callout">
            <div class="fiq-callout-label">Top Recommended Business Action</div>
            <p class="fiq-callout-text">{html.escape(insights['recommended_action'])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3 = st.columns(3, gap="medium")
    with k1:
        render_kpi_card(
            label="Most Discussed Area",
            value=insights["most_discussed_category"],
            subtitle="Highest customer mention volume",
            accent=COLORS["primary"],
            tint="#EEF2FF",
            icon_key="category",
        )
    with k2:
        render_kpi_card(
            label="Most Concerning Area",
            value=insights["most_concerning_category"],
            subtitle="Highest negative review volume",
            accent=COLORS["negative"],
            tint="#FEF2F2",
            icon_key="negative",
        )
    with k3:
        recs = insights.get("category_recommendations", [])
        urgent_cnt = sum(1 for item in recs if item["priority"] == "High")
        render_kpi_card(
            label="Actionable Categories",
            value=str(len(recs)),
            subtitle=f"{urgent_cnt} urgent (High priority)",
            accent=COLORS["secondary"],
            tint="#F0FDFA",
            icon_key="positive",
        )

    st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)

    if recs:
        for item in recs:
            priority = item["priority"]
            pill_cls = f"fiq-pill-{priority.lower()}"
            if priority == "High":
                card_cls = "fiq-rec-card fiq-rec-card-urgent"
                priority_label = "High Priority — Urgent"
            elif priority == "Medium":
                card_cls = "fiq-rec-card fiq-rec-card-medium"
                priority_label = "Medium Priority"
            else:
                card_cls = "fiq-rec-card fiq-rec-card-low"
                priority_label = "Low Priority"

            st.markdown(
                f"""
                <div class="{card_cls}">
                    <div class="fiq-rec-top">
                        <h3 class="fiq-rec-title">{html.escape(item['category'])}</h3>
                        <span class="fiq-pill {pill_cls}">{html.escape(priority_label)}</span>
                    </div>
                    <div class="fiq-rec-stats">
                        <span><strong>Negative Reviews:</strong> {item['negative_reviews']:,}</span>
                        <span><strong>Negative Feedback Rate:</strong> {item['negative_percentage']}%</span>
                    </div>
                    <p class="fiq-rec-action">
                        <strong>Recommended Action:</strong> {html.escape(item['recommendation'])}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        with st.container(border=True):
            st.markdown(
                """
                <div class="fiq-empty" style="border-color: #A7F3D0; background-color: #ECFDF5; color: #065F46;">
                    No negative feedback detected. Continue monitoring customer satisfaction.
                </div>
                """,
                unsafe_allow_html=True,
            )


# -----------------------------------
# SECTION 6: FEEDBACK EXPLORER
# -----------------------------------


def render_feedback_explorer() -> None:
    render_page_header(
        kicker="Feedback Explorer",
        title="Inspect & Export Customer Feedback",
        subtitle="Search raw customer responses, filter by sentiment or category, and export analysed CSVs.",
        source_label=source_label,
        total_records=insights["total_feedback"],
    )

    with st.container(border=True):
        f1, f2, f3 = st.columns([1.6, 1.0, 1.0], gap="medium")
        with f1:
            search_query = st.text_input(
                "Search feedback",
                placeholder="Filter by keyword or phrase...",
            )
        with f2:
            selected_sentiment = st.selectbox(
                "Sentiment",
                ["All", "Positive", "Negative", "Neutral"],
            )
        with f3:
            category_options = ["All"] + sorted(insights["category_counts"].keys())
            selected_category = st.selectbox(
                "Category",
                category_options,
            )

        explorer_df = df.copy()

        if search_query.strip():
            mask = explorer_df["feedback"].astype(str).str.contains(
                search_query.strip(), case=False, na=False, regex=False
            )
            explorer_df = explorer_df[mask]

        if selected_sentiment != "All":
            explorer_df = explorer_df[explorer_df["sentiment"] == selected_sentiment]

        if selected_category != "All":
            explorer_df = explorer_df[
                explorer_df["categories"].apply(lambda cats: selected_category in cats)
            ]

        st.caption(
            f"Showing **{len(explorer_df):,}** of **{len(df):,}** analysed feedback entries"
        )

        display_df = explorer_df.copy()
        display_df["categories"] = display_df["categories"].apply(lambda x: ", ".join(x))

        ordered_cols = [
            col
            for col in [
                "date",
                "rating",
                "feedback",
                "sentiment",
                "categories",
                "compound_score",
            ]
            if col in display_df.columns
        ]
        other_cols = [
            col
            for col in display_df.columns
            if col not in ordered_cols and col != "clean_feedback"
        ]
        display_df = display_df[ordered_cols + other_cols]

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            height=430,
            column_config={
                "feedback": st.column_config.TextColumn("Customer Feedback", width="large"),
                "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                "categories": st.column_config.TextColumn("Categories", width="medium"),
                "compound_score": st.column_config.NumberColumn(
                    "Compound Score", format="%.3f"
                ),
            },
        )

    st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        render_card_heading(
            "Export Analysis",
            "Download the full analysed dataset or the currently filtered subset as CSV",
        )
        csv_data = df.copy()
        csv_data["categories"] = csv_data["categories"].apply(lambda x: ", ".join(x))
        full_csv = csv_data.to_csv(index=False).encode("utf-8")

        e1, e2 = st.columns(2, gap="medium")
        with e1:
            st.download_button(
                label="Download Analysed Feedback CSV",
                data=full_csv,
                file_name="customer_feedback_analysis.csv",
                mime="text/csv",
                width="stretch",
            )
        with e2:
            if len(explorer_df) != len(df):
                filtered_csv_df = explorer_df.copy()
                filtered_csv_df["categories"] = filtered_csv_df["categories"].apply(
                    lambda x: ", ".join(x)
                )
                filtered_csv = filtered_csv_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"Download Filtered CSV ({len(explorer_df):,} rows)",
                    data=filtered_csv,
                    file_name="filtered_customer_feedback.csv",
                    mime="text/csv",
                    width="stretch",
                )


# -----------------------------------
# ROUTE ACTIVE SECTION
# -----------------------------------

if active_section == "Overview":
    render_overview()
elif active_section == "Sentiment Analytics":
    render_sentiment_analytics()
elif active_section == "Category Intelligence":
    render_category_intelligence()
elif active_section == "Keyword Intelligence":
    render_keyword_intelligence()
elif active_section == "Business Recommendations":
    render_business_recommendations()
elif active_section == "Feedback Explorer":
    render_feedback_explorer()