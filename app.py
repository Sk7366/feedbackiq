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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------
# DESIGN TOKENS & SAAS STYLING
# -----------------------------------

PALETTE = {
    "navy": "#0F172A",
    "navy_soft": "#1E293B",
    "bg": "#F8FAFC",
    "surface": "#FFFFFF",
    "border": "#E2E8F0",
    "text": "#0F172A",
    "muted": "#64748B",
    "indigo": "#4F46E5",
    "indigo_soft": "#EEF2FF",
    "teal": "#0D9488",
    "teal_soft": "#CCFBF1",
    "positive": "#0D9488",
    "negative": "#EF4444",
    "neutral": "#94A3B8",
    "amber": "#F59E0B",
}

SENTIMENT_COLORS = {
    "Positive": PALETTE["positive"],
    "Negative": PALETTE["negative"],
    "Neutral": PALETTE["neutral"],
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
    """Apply clean SaaS styling with deep navy sidebar and light workspace."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        /* Main light canvas */
        [data-testid="stAppViewContainer"] {
            background-color: #F8FAFC;
            color: #0F172A;
        }

        [data-testid="stHeader"] {
            background-color: rgba(248, 250, 252, 0.85);
            backdrop-filter: blur(8px);
        }

        .block-container {
            padding-top: 1.75rem;
            padding-bottom: 3rem;
            max-width: 1380px;
        }

        /* Deep navy sidebar */
        [data-testid="stSidebar"] {
            background-color: #0F172A;
            border-right: 1px solid #1E293B;
        }

        [data-testid="stSidebar"] * {
            color: #E2E8F0;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #FFFFFF !important;
        }

        [data-testid="stSidebar"] hr {
            border-color: #1E293B;
            margin: 1rem 0;
        }

        /* Sidebar navigation radio styling */
        [data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 0.3rem;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label {
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 10px;
            padding: 0.55rem 0.8rem;
            transition: all 0.15s ease;
            cursor: pointer;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background-color: #1E293B;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(90deg, rgba(79, 70, 229, 0.28), rgba(13, 148, 136, 0.16));
            border: 1px solid rgba(99, 102, 241, 0.5);
            color: #FFFFFF;
            font-weight: 600;
        }

        /* Sidebar file uploader contrast */
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background-color: #1E293B;
            border: 1px dashed #475569;
            border-radius: 10px;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
            background-color: #4F46E5;
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
        }

        /* Brand block */
        .fiq-brand {
            padding: 0.25rem 0 0.75rem 0;
        }

        .fiq-brand-row {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin-bottom: 0.45rem;
        }

        .fiq-logo-badge {
            width: 34px;
            height: 34px;
            border-radius: 10px;
            background: linear-gradient(135deg, #4F46E5 0%, #0D9488 100%);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.95rem;
            color: #FFFFFF !important;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35);
        }

        .fiq-brand-name {
            font-size: 1.2rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #FFFFFF !important;
        }

        .fiq-brand-desc {
            font-size: 0.8rem;
            line-height: 1.45;
            color: #94A3B8 !important;
            margin: 0;
        }

        .fiq-sidebar-label {
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94A3B8 !important;
            margin-bottom: 0.45rem;
        }

        .fiq-dataset-pill {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 0.65rem 0.8rem;
            font-size: 0.78rem;
            line-height: 1.45;
            margin-top: 0.5rem;
        }

        /* Page Header */
        .fiq-page-header {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: flex-end;
            gap: 1rem;
            padding-bottom: 1.15rem;
            margin-bottom: 1.35rem;
            border-bottom: 1px solid #E2E8F0;
        }

        .fiq-page-kicker {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #4F46E5;
            margin-bottom: 0.25rem;
        }

        .fiq-page-title {
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            color: #0F172A;
            margin: 0;
            line-height: 1.2;
        }

        .fiq-page-subtitle {
            font-size: 0.92rem;
            color: #64748B;
            margin: 0.3rem 0 0 0;
        }

        .fiq-status-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 999px;
            padding: 0.4rem 0.85rem;
            font-size: 0.8rem;
            font-weight: 500;
            color: #334155;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }

        .fiq-status-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background-color: #0D9488;
            display: inline-block;
        }

        /* KPI Cards */
        .fiq-kpi-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
            height: 100%;
        }

        .fiq-kpi-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.55rem;
        }

        .fiq-kpi-label {
            font-size: 0.8rem;
            font-weight: 600;
            color: #64748B;
        }

        .fiq-kpi-dot {
            width: 10px;
            height: 10px;
            border-radius: 999px;
        }

        .fiq-kpi-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #0F172A;
            letter-spacing: -0.02em;
            line-height: 1.15;
            font-variant-numeric: tabular-nums;
        }

        .fiq-kpi-sub {
            font-size: 0.78rem;
            color: #64748B;
            margin-top: 0.4rem;
        }

        /* Surface Cards */
        .fiq-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
            margin-bottom: 1rem;
        }

        .fiq-card-title {
            font-size: 1rem;
            font-weight: 600;
            color: #0F172A;
            margin: 0 0 0.2rem 0;
        }

        .fiq-card-subtitle {
            font-size: 0.82rem;
            color: #64748B;
            margin: 0 0 0.9rem 0;
        }

        /* Recent feedback items */
        .fiq-feedback-item {
            padding: 0.8rem 0.95rem;
            border: 1px solid #F1F5F9;
            background-color: #F8FAFC;
            border-radius: 10px;
            margin-bottom: 0.6rem;
        }

        .fiq-feedback-item:last-child {
            margin-bottom: 0;
        }

        .fiq-feedback-text {
            font-size: 0.86rem;
            color: #1E293B;
            line-height: 1.45;
            margin: 0 0 0.5rem 0;
        }

        .fiq-feedback-meta {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.74rem;
            color: #64748B;
        }

        /* Badges */
        .fiq-badge {
            display: inline-block;
            padding: 0.16rem 0.55rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            line-height: 1.3;
        }

        .fiq-badge-positive {
            background-color: #CCFBF1;
            color: #0F766E;
        }

        .fiq-badge-negative {
            background-color: #FEE2E2;
            color: #B91C1C;
        }

        .fiq-badge-neutral {
            background-color: #E2E8F0;
            color: #475569;
        }

        .fiq-badge-high {
            background-color: #FEE2E2;
            color: #B91C1C;
        }

        .fiq-badge-medium {
            background-color: #FEF3C7;
            color: #B45309;
        }

        .fiq-badge-low {
            background-color: #CCFBF1;
            color: #0F766E;
        }

        .fiq-cat-chip {
            background-color: #EEF2FF;
            color: #4338CA;
            padding: 0.15rem 0.5rem;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 500;
        }

        /* Recommendation cards */
        .fiq-callout {
            background: linear-gradient(135deg, #EEF2FF 0%, #F0FDFA 100%);
            border: 1px solid #C7D2FE;
            border-left: 4px solid #4F46E5;
            border-radius: 12px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1.15rem;
        }

        .fiq-callout-label {
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #4338CA;
            margin-bottom: 0.3rem;
        }

        .fiq-callout-text {
            font-size: 0.98rem;
            font-weight: 600;
            color: #0F172A;
            margin: 0;
            line-height: 1.45;
        }

        .fiq-rec-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
            margin-bottom: 0.85rem;
        }

        .fiq-rec-header {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.55rem;
        }

        .fiq-rec-title {
            font-size: 1.02rem;
            font-weight: 600;
            color: #0F172A;
            margin: 0;
        }

        .fiq-rec-metrics {
            display: flex;
            gap: 1.25rem;
            font-size: 0.8rem;
            color: #475569;
            margin-bottom: 0.6rem;
        }

        .fiq-rec-body {
            font-size: 0.9rem;
            color: #1E293B;
            line-height: 1.5;
            margin: 0;
            padding-top: 0.55rem;
            border-top: 1px solid #F1F5F9;
        }

        /* Empty state */
        .fiq-empty-state {
            background-color: #FFFFFF;
            border: 1px dashed #CBD5E1;
            border-radius: 14px;
            padding: 1.75rem 1.25rem;
            text-align: center;
            color: #64748B;
            font-size: 0.88rem;
        }

        /* Streamlit native button polish */
        .stDownloadButton button {
            background-color: #4F46E5;
            color: #FFFFFF;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.55rem 1.1rem;
        }

        .stDownloadButton button:hover {
            background-color: #4338CA;
            color: #FFFFFF;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_plotly_figure(fig: go.Figure, height: int = 340) -> go.Figure:
    """Apply consistent SaaS typography, light surface, and subtle gridlines."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            color="#334155",
            size=12,
        ),
        margin=dict(l=12, r=16, t=28, b=16),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text="",
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_size=12,
            font_color="#FFFFFF",
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#E2E8F0",
        tickfont=dict(color="#64748B"),
        title_font=dict(color="#475569", size=12),
    )
    fig.update_yaxes(
        gridcolor="#F1F5F9",
        zerolinecolor="#E2E8F0",
        tickfont=dict(color="#64748B"),
        title_font=dict(color="#475569", size=12),
    )
    return fig


def render_kpi_card(
    label: str,
    value: str,
    subtitle: str,
    accent: str = PALETTE["indigo"],
) -> None:
    """Render a styled SaaS KPI metric card."""
    st.markdown(
        f"""
        <div class="fiq-kpi-card">
            <div class="fiq-kpi-top">
                <span class="fiq-kpi-label">{html.escape(label)}</span>
                <span class="fiq-kpi-dot" style="background-color: {accent};"></span>
            </div>
            <div class="fiq-kpi-value">{html.escape(str(value))}</div>
            <div class="fiq-kpi-sub">{html.escape(subtitle)}</div>
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
    """Render the top SaaS section header and dataset status chip."""
    st.markdown(
        f"""
        <div class="fiq-page-header">
            <div>
                <div class="fiq-page-kicker">{html.escape(kicker)}</div>
                <h1 class="fiq-page-title">{html.escape(title)}</h1>
                <p class="fiq-page-subtitle">{html.escape(subtitle)}</p>
            </div>
            <div class="fiq-status-chip">
                <span class="fiq-status-dot"></span>
                <span>{html.escape(source_label)} &middot; {total_records:,} records</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_valid_rating_df(df: pd.DataFrame) -> pd.DataFrame:
    """Safely extract numeric rating rows if a rating column exists."""
    if "rating" not in df.columns:
        return pd.DataFrame()
    rating_df = df.copy()
    rating_df["rating"] = pd.to_numeric(rating_df["rating"], errors="coerce")
    return rating_df.dropna(subset=["rating"])


def get_valid_date_df(df: pd.DataFrame) -> pd.DataFrame:
    """Safely extract valid parsed date rows if a date column exists."""
    if "date" not in df.columns:
        return pd.DataFrame()
    trend_df = df.copy()
    trend_df["date"] = pd.to_datetime(trend_df["date"], errors="coerce")
    return trend_df.dropna(subset=["date"])


def build_sentiment_summary_df(insights: dict) -> pd.DataFrame:
    """Build an ordered sentiment DataFrame covering all three classes."""
    counts = insights["sentiment_counts"]
    percentages = insights["sentiment_percentages"]
    rows = []
    for label in ["Positive", "Neutral", "Negative"]:
        rows.append(
            {
                "Sentiment": label,
                "Count": int(counts.get(label, 0)),
                "Percentage": percentages.get(label, 0.0),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------
# CACHED NLP PIPELINE
# -----------------------------------


@st.cache_data(show_spinner=False)
def run_analysis_pipeline(raw_df: pd.DataFrame):
    """Run existing preprocessing, VADER sentiment, categorization, and insights."""
    # Prevent duplicate columns if an already-analysed CSV export is re-uploaded
    computed_cols = [
        "clean_feedback",
        "positive_score",
        "negative_score",
        "neutral_score",
        "compound_score",
        "sentiment",
        "categories",
    ]
    clean_input = raw_df.drop(
        columns=[c for c in computed_cols if c in raw_df.columns]
    )

    processed_df = preprocess_feedback(clean_input)
    if processed_df.empty:
        return processed_df, None

    analysed_df = analyze_dataset(processed_df)
    analysed_df = add_categories(analysed_df)
    insights = generate_business_insights(analysed_df)
    return analysed_df, insights


# -----------------------------------
# INITIALIZE STYLING
# -----------------------------------

inject_custom_css()


# -----------------------------------
# A. SIDEBAR (BRANDING, DATA, NAV)
# -----------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="fiq-brand">
            <div class="fiq-brand-row">
                <span class="fiq-logo-badge">IQ</span>
                <span class="fiq-brand-name">FeedbackIQ</span>
            </div>
            <p class="fiq-brand-desc">
                AI-powered customer feedback intelligence for sentiment,
                theme detection, and prioritized actions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown('<div class="fiq-sidebar-label">Navigation</div>', unsafe_allow_html=True)
    active_section = st.radio(
        "Dashboard Section",
        NAV_SECTIONS,
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown('<div class="fiq-sidebar-label">Data Source</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload customer feedback CSV",
        type=["csv"],
        help="CSV must include a 'feedback' column. Optional: 'date' and 'rating'.",
    )

    use_sample = st.checkbox(
        "Use sample feedback dataset",
        value=uploaded_file is None,
    )


# -----------------------------------
# LOAD & PROCESS DATASET
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
        st.markdown(
            """
            <div class="fiq-card" style="text-align:center; padding: 2.5rem 1.5rem;">
                <div class="fiq-page-kicker">FeedbackIQ Workspace</div>
                <h2 style="margin: 0.25rem 0 0.5rem 0; color: #0F172A;">
                    Upload a CSV or enable the sample dataset
                </h2>
                <p style="color: #64748B; max-width: 520px; margin: 0 auto;">
                    Use the sidebar to upload your customer feedback CSV
                    (requiring a <code>feedback</code> column) or select
                    <strong>Use sample feedback dataset</strong> to explore the dashboard.
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
        df, insights = run_analysis_pipeline(raw_df)

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


# Sidebar active dataset summary
rating_df_global = get_valid_rating_df(df)
date_df_global = get_valid_date_df(df)

with st.sidebar:
    optional_signals = []
    if not date_df_global.empty:
        optional_signals.append("Dates")
    if not rating_df_global.empty:
        optional_signals.append("Ratings")
    signals_text = ", ".join(optional_signals) if optional_signals else "Text only"

    st.markdown(
        f"""
        <div class="fiq-dataset-pill">
            <strong>Active:</strong> {html.escape(source_label)}<br/>
            <strong>Analysed:</strong> {insights['total_feedback']:,} entries<br/>
            <strong>Signals:</strong> {html.escape(signals_text)}
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------
# SECTION RENDERERS
# -----------------------------------


def render_overview() -> None:
    render_page_header(
        kicker="Executive Summary",
        title="Customer Feedback Overview",
        subtitle="Real-time health metrics, sentiment mix, and latest customer voices.",
        source_label=source_label,
        total_records=insights["total_feedback"],
    )

    total = insights["total_feedback"]
    counts = insights["sentiment_counts"]
    percentages = insights["sentiment_percentages"]

    # 4-Column KPI Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card(
            label="Total Feedback Analysed",
            value=f"{total:,}",
            subtitle=f"{len(insights['category_counts'])} feedback categories detected",
            accent=PALETTE["indigo"],
        )
    with k2:
        render_kpi_card(
            label="Positive Feedback",
            value=f"{percentages['Positive']}%",
            subtitle=f"{counts.get('Positive', 0):,} positive responses",
            accent=PALETTE["positive"],
        )
    with k3:
        render_kpi_card(
            label="Negative Feedback",
            value=f"{percentages['Negative']}%",
            subtitle=f"{counts.get('Negative', 0):,} negative responses",
            accent=PALETTE["negative"],
        )
    with k4:
        if not rating_df_global.empty:
            avg_rating = rating_df_global["rating"].mean()
            render_kpi_card(
                label="Average Customer Rating",
                value=f"{avg_rating:.2f} / 5",
                subtitle=f"Based on {len(rating_df_global):,} rated entries",
                accent=PALETTE["amber"],
            )
        else:
            render_kpi_card(
                label="Neutral Feedback",
                value=f"{percentages['Neutral']}%",
                subtitle=f"{counts.get('Neutral', 0):,} neutral responses",
                accent=PALETTE["neutral"],
            )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Main 2-column grid: Sentiment Distribution + Recent Feedback
    left_col, right_col = st.columns([1.05, 1.25], gap="large")

    with left_col:
        st.markdown(
            """
            <div class="fiq-card">
                <div class="fiq-card-title">Sentiment Distribution</div>
                <div class="fiq-card-subtitle">Overall proportion of Positive, Neutral, and Negative feedback</div>
            """,
            unsafe_allow_html=True,
        )

        sentiment_df = build_sentiment_summary_df(insights)
        fig = px.pie(
            sentiment_df,
            names="Sentiment",
            values="Count",
            hole=0.62,
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
        )
        fig.update_traces(
            textposition="outside",
            textinfo="percent+label",
            marker=dict(line=dict(color="#FFFFFF", width=2)),
        )
        fig.add_annotation(
            text=f"<b>{total:,}</b><br><span style='font-size:11px;color:#64748B;'>Reviews</span>",
            x=0.5,
            y=0.5,
            font=dict(size=18, color="#0F172A"),
            showarrow=False,
        )
        style_plotly_figure(fig, height=340)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown(
            """
            <div class="fiq-card">
                <div class="fiq-card-title">Recent Feedback Overview</div>
                <div class="fiq-card-subtitle">Most recent customer comments with detected sentiment and topics</div>
            """,
            unsafe_allow_html=True,
        )

        if not date_df_global.empty:
            recent_df = date_df_global.sort_values("date", ascending=False).head(5)
        else:
            recent_df = df.tail(5).iloc[::-1]

        items_html = []
        for _, row in recent_df.iterrows():
            raw_text = str(row["feedback"])
            snippet = raw_text if len(raw_text) <= 170 else raw_text[:167].rstrip() + "..."
            sentiment = str(row["sentiment"])
            badge_class = f"fiq-badge-{sentiment.lower()}"
            cats = row.get("categories", ["Other"])
            cat_chips = "".join(
                f'<span class="fiq-cat-chip">{html.escape(str(c))}</span>'
                for c in cats
            )

            extra_meta = []
            if "date" in row and pd.notna(row["date"]):
                date_str = (
                    row["date"].strftime("%Y-%m-%d")
                    if hasattr(row["date"], "strftime")
                    else str(row["date"])
                )
                extra_meta.append(html.escape(date_str))
            if "rating" in row and pd.notna(row["rating"]):
                try:
                    extra_meta.append(f"Rating: {float(row['rating']):.1f}/5")
                except (TypeError, ValueError):
                    pass

            meta_suffix = (
                f"<span>{' &middot; '.join(extra_meta)}</span>" if extra_meta else ""
            )

            items_html.append(
                f"""
                <div class="fiq-feedback-item">
                    <p class="fiq-feedback-text">{html.escape(snippet)}</p>
                    <div class="fiq-feedback-meta">
                        <span class="fiq-badge {badge_class}">{html.escape(sentiment)}</span>
                        {cat_chips}
                        {meta_suffix}
                    </div>
                </div>
                """
            )

        st.markdown("".join(items_html) + "</div>", unsafe_allow_html=True)

    # Bottom Executive Highlights
    h1, h2, h3 = st.columns(3)
    with h1:
        render_kpi_card(
            label="Most Discussed Category",
            value=insights["most_discussed_category"],
            subtitle=f"{insights['category_counts'].get(insights['most_discussed_category'], 0)} total mentions",
            accent=PALETTE["indigo"],
        )
    with h2:
        concerning = insights["most_concerning_category"]
        concerning_neg = insights["negative_category_counts"].get(concerning, 0)
        render_kpi_card(
            label="Most Concerning Category",
            value=concerning,
            subtitle=(
                f"{concerning_neg} negative reviews"
                if concerning != "None"
                else "No category meets minimum threshold"
            ),
            accent=PALETTE["negative"],
        )
    with h3:
        top_kw = insights["top_keywords"][0][0] if insights["top_keywords"] else "N/A"
        top_kw_count = insights["top_keywords"][0][1] if insights["top_keywords"] else 0
        render_kpi_card(
            label="Top Recurring Keyword",
            value=top_kw.title() if top_kw != "N/A" else "N/A",
            subtitle=f"Mentioned {top_kw_count} times",
            accent=PALETTE["teal"],
        )


def render_sentiment_analytics() -> None:
    render_page_header(
        kicker="Sentiment Analytics",
        title="Customer Sentiment & Rating Intelligence",
        subtitle="Inspect polarity breakdown, daily sentiment trajectory, and rating alignment.",
        source_label=source_label,
        total_records=insights["total_feedback"],
    )

    counts = insights["sentiment_counts"]
    percentages = insights["sentiment_percentages"]
    avg_compound = df["compound_score"].mean()

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        render_kpi_card(
            label="Positive Reviews",
            value=f"{counts.get('Positive', 0):,}",
            subtitle=f"{percentages['Positive']}% of total",
            accent=PALETTE["positive"],
        )
    with s2:
        render_kpi_card(
            label="Neutral Reviews",
            value=f"{counts.get('Neutral', 0):,}",
            subtitle=f"{percentages['Neutral']}% of total",
            accent=PALETTE["neutral"],
        )
    with s3:
        render_kpi_card(
            label="Negative Reviews",
            value=f"{counts.get('Negative', 0):,}",
            subtitle=f"{percentages['Negative']}% of total",
            accent=PALETTE["negative"],
        )
    with s4:
        render_kpi_card(
            label="Mean VADER Compound",
            value=f"{avg_compound:+.3f}",
            subtitle="Scale from -1.000 to +1.000",
            accent=PALETTE["indigo"],
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    sentiment_df = build_sentiment_summary_df(insights)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <div class="fiq-card">
                <div class="fiq-card-title">Sentiment Distribution</div>
                <div class="fiq-card-subtitle">Share of customer feedback by sentiment class</div>
            """,
            unsafe_allow_html=True,
        )
        fig_pie = px.pie(
            sentiment_df,
            names="Sentiment",
            values="Count",
            hole=0.58,
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
        )
        fig_pie.update_traces(
            textinfo="percent+label",
            marker=dict(line=dict(color="#FFFFFF", width=2)),
        )
        style_plotly_figure(fig_pie, height=320)
        fig_pie.update_layout(showlegend=False)
        st.plotly_chart(fig_pie, width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            <div class="fiq-card">
                <div class="fiq-card-title">Sentiment Volume</div>
                <div class="fiq-card-subtitle">Absolute review counts across sentiment classifications</div>
            """,
            unsafe_allow_html=True,
        )
        fig_bar = px.bar(
            sentiment_df,
            x="Sentiment",
            y="Count",
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            text="Count",
        )
        fig_bar.update_traces(textposition="outside", marker_line_width=0)
        style_plotly_figure(fig_bar, height=320)
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # Sentiment Trends Over Time
    st.markdown(
        """
        <div class="fiq-card">
            <div class="fiq-card-title">Sentiment Trends Over Time</div>
            <div class="fiq-card-subtitle">Daily volume of Positive, Neutral, and Negative feedback</div>
        """,
        unsafe_allow_html=True,
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
            color_discrete_map=SENTIMENT_COLORS,
        )
        fig_trend.update_traces(line=dict(width=2.5), marker=dict(size=7))
        style_plotly_figure(fig_trend, height=330)
        fig_trend.update_layout(xaxis_title="Date", yaxis_title="Feedback Count")
        st.plotly_chart(fig_trend, width="stretch", config={"displayModeBar": False})
    else:
        st.markdown(
            """
            <div class="fiq-empty-state">
                No valid <code>date</code> column detected in this dataset.
                Include a parseable <code>date</code> column to view daily sentiment trends.
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Rating vs Sentiment Analysis
    st.markdown(
        """
        <div class="fiq-card">
            <div class="fiq-card-title">Rating vs. Sentiment Analysis</div>
            <div class="fiq-card-subtitle">How numeric customer ratings align with predicted NLP sentiment</div>
        """,
        unsafe_allow_html=True,
    )
    if not rating_df_global.empty:
        r_col1, r_col2 = st.columns(2, gap="large")
        with r_col1:
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
                color_discrete_sequence=[PALETTE["indigo"]],
                title="Customer Rating Distribution",
            )
            fig_rating.update_traces(textposition="outside")
            style_plotly_figure(fig_rating, height=310)
            st.plotly_chart(fig_rating, width="stretch", config={"displayModeBar": False})

        with r_col2:
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
                color_discrete_map=SENTIMENT_COLORS,
                text_auto=".2f",
                title="Average Rating by Predicted Sentiment",
            )
            fig_sr.update_traces(textposition="outside")
            style_plotly_figure(fig_sr, height=310)
            fig_sr.update_layout(
                showlegend=False,
                xaxis_title="Predicted Sentiment",
                yaxis_title="Average Rating",
                yaxis=dict(range=[0, 5.6]),
            )
            st.plotly_chart(fig_sr, width="stretch", config={"displayModeBar": False})
    else:
        st.markdown(
            """
            <div class="fiq-empty-state">
                No valid <code>rating</code> column detected in this dataset.
                Include numeric ratings (e.g., 1–5) to enable rating vs. sentiment analysis.
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_category_intelligence() -> None:
    render_page_header(
        kicker="Category Intelligence",
        title="Theme & Priority Analysis",
        subtitle="Pinpoint high-volume discussion drivers and high-risk operational categories.",
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

    high_count = int((priority_df["Priority"] == "High").sum()) if not priority_df.empty else 0
    med_count = int((priority_df["Priority"] == "Medium").sum()) if not priority_df.empty else 0
    low_count = int((priority_df["Priority"] == "Low").sum()) if not priority_df.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            label="Most Discussed Category",
            value=insights["most_discussed_category"],
            subtitle=f"{insights['category_counts'].get(insights['most_discussed_category'], 0)} mentions",
            accent=PALETTE["indigo"],
        )
    with c2:
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
            accent=PALETTE["negative"],
        )
    with c3:
        render_kpi_card(
            label="High Priority Areas",
            value=str(high_count),
            subtitle=f"{med_count} medium priority areas",
            accent=PALETTE["negative"],
        )
    with c4:
        render_kpi_card(
            label="Low Priority Areas",
            value=str(low_count),
            subtitle="Within healthy thresholds",
            accent=PALETTE["teal"],
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <div class="fiq-card">
                <div class="fiq-card-title">Feedback Category Distribution</div>
                <div class="fiq-card-subtitle">Total customer mentions across business areas</div>
            """,
            unsafe_allow_html=True,
        )
        category_data = pd.DataFrame(
            list(insights["category_counts"].items()),
            columns=["Category", "Feedback Count"],
        )
        if not category_data.empty:
            fig_cat = px.bar(
                category_data.sort_values("Feedback Count", ascending=True),
                x="Feedback Count",
                y="Category",
                orientation="h",
                text="Feedback Count",
                color_discrete_sequence=[PALETTE["indigo"]],
            )
            fig_cat.update_traces(textposition="outside")
            style_plotly_figure(fig_cat, height=340)
            st.plotly_chart(fig_cat, width="stretch", config={"displayModeBar": False})
        else:
            st.info("No categories available.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            <div class="fiq-card">
                <div class="fiq-card-title">Category-wise Negative Feedback Rate</div>
                <div class="fiq-card-subtitle">Percentage of negative feedback within each category</div>
            """,
            unsafe_allow_html=True,
        )
        negative_rates = pd.DataFrame(
            list(insights["category_negative_rates"].items()),
            columns=["Category", "Negative Rate (%)"],
        )
        if not negative_rates.empty:
            fig_neg = px.bar(
                negative_rates.sort_values("Negative Rate (%)", ascending=True),
                x="Negative Rate (%)",
                y="Category",
                orientation="h",
                text_auto=".1f",
                color="Negative Rate (%)",
                color_continuous_scale=["#0D9488", "#F59E0B", "#EF4444"],
            )
            fig_neg.update_traces(textposition="outside")
            style_plotly_figure(fig_neg, height=340)
            fig_neg.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_neg, width="stretch", config={"displayModeBar": False})
        else:
            st.info("No negative rate data available.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Category Priority Matrix Table
    st.markdown(
        """
        <div class="fiq-card">
            <div class="fiq-card-title">Category Priority Matrix</div>
            <div class="fiq-card-subtitle">Business areas ranked by negative review volume and negative rate</div>
        """,
        unsafe_allow_html=True,
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
                "Negative Rate (%)": st.column_config.ProgressColumn(
                    "Negative Rate",
                    min_value=0,
                    max_value=100,
                    format="%.2f%%",
                ),
            },
        )
    else:
        st.info("No category priority data available.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_keyword_intelligence() -> None:
    render_page_header(
        kicker="Keyword Intelligence",
        title="Recurring Terms & Language Signals",
        subtitle="Discover the most frequently mentioned terms across customer feedback.",
        source_label=source_label,
        total_records=insights["total_feedback"],
    )

    top_n = st.slider(
        "Number of top keywords to display",
        min_value=5,
        max_value=25,
        value=10,
        step=5,
    )

    keywords = get_top_keywords(df, top_n=top_n)
    keyword_df = pd.DataFrame(keywords, columns=["Keyword", "Frequency"])

    if keyword_df.empty:
        st.markdown(
            '<div class="fiq-empty-state">No meaningful keywords available for analysis.</div>',
            unsafe_allow_html=True,
        )
        return

    total_mentions = int(keyword_df["Frequency"].sum())
    keyword_df["Share (%)"] = (keyword_df["Frequency"] / total_mentions * 100).round(1)

    col1, col2 = st.columns([1.15, 1.0], gap="large")

    with col1:
        st.markdown(
            """
            <div class="fiq-card">
                <div class="fiq-card-title">Most Frequently Occurring Keywords</div>
                <div class="fiq-card-subtitle">Ranked term frequency after stopword filtering</div>
            """,
            unsafe_allow_html=True,
        )
        fig_kw = px.bar(
            keyword_df.sort_values("Frequency", ascending=True),
            x="Frequency",
            y="Keyword",
            orientation="h",
            text="Frequency",
            color="Frequency",
            color_continuous_scale=["#99F6E4", "#0D9488", "#4F46E5"],
        )
        fig_kw.update_traces(textposition="outside")
        style_plotly_figure(fig_kw, height=max(340, top_n * 26))
        fig_kw.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_kw, width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            <div class="fiq-card">
                <div class="fiq-card-title">Keyword Density Map</div>
                <div class="fiq-card-subtitle">Proportional visual breakdown of dominant customer terms</div>
            """,
            unsafe_allow_html=True,
        )
        fig_tree = px.treemap(
            keyword_df,
            path=["Keyword"],
            values="Frequency",
            color="Frequency",
            color_continuous_scale=["#CCFBF1", "#0D9488", "#4F46E5"],
        )
        fig_tree.update_traces(
            textinfo="label+value",
            hovertemplate="<b>%{label}</b><br>Mentions: %{value}<extra></extra>",
        )
        style_plotly_figure(fig_tree, height=340)
        fig_tree.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=4, r=4, t=8, b=4),
        )
        st.plotly_chart(fig_tree, width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="fiq-card">
                <div class="fiq-card-title">Keyword Frequency Table</div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            keyword_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Share (%)": st.column_config.ProgressColumn(
                    "Share of Top Terms",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                )
            },
        )
        st.markdown("</div>", unsafe_allow_html=True)


def render_business_recommendations() -> None:
    render_page_header(
        kicker="Business Recommendations",
        title="Actionable Improvement Roadmap",
        subtitle="Prioritized strategic actions derived from category-level negative feedback.",
        source_label=source_label,
        total_records=insights["total_feedback"],
    )

    # Top Recommended Action Callout
    st.markdown(
        f"""
        <div class="fiq-callout">
            <div class="fiq-callout-label">Top Recommended Business Action</div>
            <p class="fiq-callout-text">{html.escape(insights['recommended_action'])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key Findings Row
    f1, f2, f3 = st.columns(3)
    with f1:
        render_kpi_card(
            label="Most Discussed Area",
            value=insights["most_discussed_category"],
            subtitle="Highest customer mention volume",
            accent=PALETTE["indigo"],
        )
    with f2:
        render_kpi_card(
            label="Most Concerning Area",
            value=insights["most_concerning_category"],
            subtitle="Highest negative review impact",
            accent=PALETTE["negative"],
        )
    with f3:
        rec_count = len(insights.get("category_recommendations", []))
        render_kpi_card(
            label="Active Action Items",
            value=str(rec_count),
            subtitle="Categories with negative feedback",
            accent=PALETTE["teal"],
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    category_recommendations = insights.get("category_recommendations", [])

    if category_recommendations:
        for item in category_recommendations:
            priority = item["priority"]
            badge_class = f"fiq-badge-{priority.lower()}"
            st.markdown(
                f"""
                <div class="fiq-rec-card">
                    <div class="fiq-rec-header">
                        <h3 class="fiq-rec-title">{html.escape(item['category'])}</h3>
                        <span class="fiq-badge {badge_class}">{html.escape(priority)} Priority</span>
                    </div>
                    <div class="fiq-rec-metrics">
                        <span><strong>Negative Reviews:</strong> {item['negative_reviews']}</span>
                        <span><strong>Negative Feedback Rate:</strong> {item['negative_percentage']}%</span>
                    </div>
                    <p class="fiq-rec-body">
                        <strong>Recommended Action:</strong> {html.escape(item['recommendation'])}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.success(
            "No negative feedback detected. Continue monitoring customer satisfaction."
        )


def render_feedback_explorer() -> None:
    render_page_header(
        kicker="Feedback Explorer",
        title="Search, Filter & Export Feedback",
        subtitle="Inspect individual customer entries, verify sentiment labels, and download CSV reports.",
        source_label=source_label,
        total_records=insights["total_feedback"],
    )

    # Filter toolbar
    f_col1, f_col2, f_col3 = st.columns([1.6, 1.0, 1.0])

    with f_col1:
        search_query = st.text_input(
            "Search feedback by keyword",
            placeholder="e.g. delivery, refund, broken, checkout...",
        )

    with f_col2:
        selected_sentiment = st.selectbox(
            "Filter by sentiment",
            ["All", "Positive", "Negative", "Neutral"],
        )

    with f_col3:
        category_options = ["All"] + sorted(insights["category_counts"].keys())
        selected_category = st.selectbox(
            "Filter by category",
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

    st.caption(f"Showing **{len(explorer_df):,}** of **{len(df):,}** analysed entries")

    # Format display DataFrame cleanly without mutating df
    display_df = explorer_df.copy()
    display_df["categories"] = display_df["categories"].apply(lambda x: ", ".join(x))

    preferred_order = [
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
    remaining_cols = [
        col
        for col in display_df.columns
        if col not in preferred_order and col != "clean_feedback"
    ]
    display_df = display_df[preferred_order + remaining_cols]

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        height=440,
        column_config={
            "feedback": st.column_config.TextColumn("Customer Feedback", width="large"),
            "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
            "categories": st.column_config.TextColumn("Categories", width="medium"),
            "compound_score": st.column_config.NumberColumn(
                "Compound Score", format="%.3f"
            ),
        },
    )

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # Export section (preserves full analysed CSV download)
    st.markdown(
        """
        <div class="fiq-card">
            <div class="fiq-card-title">Export Analysed Data</div>
            <div class="fiq-card-subtitle">Download complete NLP sentiment and category results as CSV</div>
        """,
        unsafe_allow_html=True,
    )

    csv_data = df.copy()
    csv_data["categories"] = csv_data["categories"].apply(lambda x: ", ".join(x))
    full_download = csv_data.to_csv(index=False).encode("utf-8")

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.download_button(
            label="Download Analysed Feedback CSV",
            data=full_download,
            file_name="customer_feedback_analysis.csv",
            mime="text/csv",
            width="stretch",
        )

    with exp_col2:
        if len(explorer_df) != len(df):
            filtered_csv = explorer_df.copy()
            filtered_csv["categories"] = filtered_csv["categories"].apply(
                lambda x: ", ".join(x)
            )
            filtered_download = filtered_csv.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download Filtered View ({len(explorer_df):,} rows)",
                data=filtered_download,
                file_name="filtered_customer_feedback.csv",
                mime="text/csv",
                width="stretch",
            )

    st.markdown("</div>", unsafe_allow_html=True)


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