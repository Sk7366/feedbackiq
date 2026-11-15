
import streamlit as st
import pandas as pd
import plotly.express as px

from src.preprocessing import preprocess_feedback
from src.sentiment import analyze_dataset
from src.insights import (
    add_categories,
    get_top_keywords,
    generate_business_insights
)


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Customer Feedback Intelligence",
    page_icon="📊",
    layout="wide"
)


# -----------------------------------
# CUSTOM STYLING
# -----------------------------------

st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }

    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }

    h1, h2, h3 {
        color: #172554;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------
# HEADER
# -----------------------------------

st.title("Customer Feedback Intelligence")

st.markdown(
    "### Transform customer feedback into actionable business insights."
)

st.caption(
    "Upload customer feedback and discover sentiment, "
    "recurring issues and customer satisfaction trends."
)

st.divider()


# -----------------------------------
# SIDEBAR - CSV UPLOAD
# -----------------------------------

st.sidebar.title("Data Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload customer feedback CSV",
    type=["csv"]
)

use_sample = st.sidebar.checkbox(
    "Use sample feedback dataset",
    value=uploaded_file is None
)


# -----------------------------------
# LOAD DATA
# -----------------------------------

df = None

try:

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

    elif use_sample:
        df = pd.read_csv("data/sample_feedback.csv")

    else:
        st.info("Upload a CSV file to begin analysis.")
        st.stop()

    # Validate feedback column
    if "feedback" not in df.columns:
        st.error(
            "Your CSV must contain a column named 'feedback'."
        )
        st.stop()

    # -----------------------------------
    # NLP PROCESSING PIPELINE
    # -----------------------------------

    with st.spinner("Analysing customer feedback..."):

        df = preprocess_feedback(df)

        if df.empty:
            st.error("No valid customer feedback found.")
            st.stop()

        df = analyze_dataset(df)

        df = add_categories(df)

        insights = generate_business_insights(df)


except Exception as e:

    st.error(f"Unable to process the dataset: {e}")
    st.stop()


# -----------------------------------
# OVERVIEW METRICS
# -----------------------------------

st.header("Business Overview")

col1, col2, col3, col4 = st.columns(4)

total = insights["total_feedback"]

percentages = insights["sentiment_percentages"]

with col1:
    st.metric(
        "Total Feedback",
        total
    )

with col2:
    st.metric(
        "Positive Feedback",
        f"{percentages['Positive']}%"
    )

with col3:
    st.metric(
        "Negative Feedback",
        f"{percentages['Negative']}%"
    )

with col4:
    st.metric(
        "Neutral Feedback",
        f"{percentages['Neutral']}%"
    )

st.divider()


# -----------------------------------
# SENTIMENT ANALYSIS
# -----------------------------------

st.header("Sentiment Analysis")

col1, col2 = st.columns(2)

sentiment_data = pd.DataFrame({
    "Sentiment": list(
        insights["sentiment_counts"].keys()
    ),
    "Count": list(
        insights["sentiment_counts"].values()
    )
})

with col1:

    st.subheader("Sentiment Distribution")

    fig = px.pie(
        sentiment_data,
        names="Sentiment",
        values="Count",
        hole=0.5,
        color="Sentiment",
        color_discrete_map={
            "Positive": "#22c55e",
            "Negative": "#ef4444",
            "Neutral": "#94a3b8"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    st.subheader("Sentiment Volume")

    fig = px.bar(
        sentiment_data,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        color_discrete_map={
            "Positive": "#22c55e",
            "Negative": "#ef4444",
            "Neutral": "#94a3b8"
        },
        text="Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================
# ADVANCED ANALYTICS
# ==========================================

st.divider()

st.header("Advanced Customer Analytics")


# -----------------------------------
# 1. SENTIMENT TRENDS OVER TIME
# -----------------------------------

if "date" in df.columns:

    st.subheader("Sentiment Trends Over Time")

    trend_df = df.copy()

    trend_df["date"] = pd.to_datetime(
        trend_df["date"],
        errors="coerce"
    )

    trend_df = trend_df.dropna(subset=["date"])

    if not trend_df.empty:

        trend_df["date"] = trend_df["date"].dt.date

        daily_sentiment = (
            trend_df.groupby(["date", "sentiment"])
            .size()
            .reset_index(name="Feedback Count")
        )

        fig = px.line(
            daily_sentiment,
            x="date",
            y="Feedback Count",
            color="sentiment",
            markers=True,
            title="Daily Customer Sentiment Trends"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.info("No valid dates available for trend analysis.")


# -----------------------------------
# 2. CUSTOMER RATING ANALYSIS
# -----------------------------------

if "rating" in df.columns:

    st.subheader("Customer Rating Analysis")

    rating_df = df.copy()

    rating_df["rating"] = pd.to_numeric(
        rating_df["rating"],
        errors="coerce"
    )

    rating_df = rating_df.dropna(subset=["rating"])

    if not rating_df.empty:

        col1, col2 = st.columns(2)

        with col1:

            rating_counts = (
                rating_df["rating"]
                .value_counts()
                .sort_index()
                .reset_index()
            )

            rating_counts.columns = ["Rating", "Count"]

            fig = px.bar(
                rating_counts,
                x="Rating",
                y="Count",
                text="Count",
                title="Customer Rating Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            average_rating = rating_df["rating"].mean()

            st.metric(
                "Average Customer Rating",
                f"{average_rating:.2f} / 5"
            )

            sentiment_rating = (
                rating_df.groupby("sentiment")["rating"]
                .mean()
                .reset_index()
            )

            fig = px.bar(
                sentiment_rating,
                x="sentiment",
                y="rating",
                text_auto=".2f",
                title="Average Rating by Predicted Sentiment"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:
        st.info("No valid ratings available for analysis.")


st.divider()


# -----------------------------------
# CATEGORY ANALYSIS
# -----------------------------------

st.header("Feedback Category Analysis")

category_data = pd.DataFrame(
    list(insights["category_counts"].items()),
    columns=["Category", "Feedback Count"]
)

if not category_data.empty:

    fig = px.bar(
        category_data.sort_values(
            "Feedback Count",
            ascending=True
        ),
        x="Feedback Count",
        y="Category",
        orientation="h",
        text="Feedback Count",
        title="Customer Feedback by Category"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


col1, col2 = st.columns(2)

with col1:

    st.subheader("Most Discussed Category")

    st.info(
        insights["most_discussed_category"]
    )

with col2:

    st.subheader("Most Concerning Category")

    st.warning(
        insights["most_concerning_category"]
    )


# -----------------------------------
# NEGATIVE FEEDBACK RATES
# -----------------------------------

negative_rates = pd.DataFrame(
    list(
        insights["category_negative_rates"].items()
    ),
    columns=["Category", "Negative Rate (%)"]
)

if not negative_rates.empty:

    st.subheader("Category-wise Negative Feedback Rate")

    st.plotly_chart(
        px.bar(
            negative_rates.sort_values(
                "Negative Rate (%)",
                ascending=True
            ),
            x="Negative Rate (%)",
            y="Category",
            orientation="h",
            title="Percentage of Negative Feedback by Category"
        ),
        use_container_width=True
    )


# -----------------------------------
# CATEGORY PRIORITY ANALYSIS
# -----------------------------------

st.divider()

st.header("Category Priority Analysis")

st.caption(
    "Identify which business areas need attention "
    "based on negative feedback volume and percentage."
)

category_analysis = insights["category_analysis"]

if category_analysis:

    priority_df = pd.DataFrame([
        {
            "Category": category,
            "Total Reviews": metrics["total_reviews"],
            "Negative Reviews": metrics["negative_reviews"],
            "Negative Rate (%)": metrics["negative_percentage"],
            "Priority": metrics["priority"]
        }
        for category, metrics in category_analysis.items()
    ])

    # Sort by negative review volume
    priority_df = priority_df.sort_values(
        by=["Negative Reviews", "Negative Rate (%)"],
        ascending=False
    )

    st.dataframe(
        priority_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Negative Rate (%)": st.column_config.ProgressColumn(
                "Negative Rate",
                min_value=0,
                max_value=100,
                format="%.2f%%"
            )
        }
    )

    # Priority summary

    high = (
        priority_df["Priority"] == "High"
    ).sum()

    medium = (
        priority_df["Priority"] == "Medium"
    ).sum()

    low = (
        priority_df["Priority"] == "Low"
    ).sum()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "High Priority Categories",
        int(high)
    )

    col2.metric(
        "Medium Priority Categories",
        int(medium)
    )

    col3.metric(
        "Low Priority Categories",
        int(low)
    )

else:

    st.info("No category priority data available.")


# -----------------------------------
# KEYWORD INTELLIGENCE
# -----------------------------------

st.divider()

st.header("Keyword Intelligence")

keywords = get_top_keywords(
    df,
    top_n=10
)

keyword_df = pd.DataFrame(
    keywords,
    columns=["Keyword", "Frequency"]
)

if not keyword_df.empty:

    st.plotly_chart(
        px.bar(
            keyword_df.sort_values(
                "Frequency",
                ascending=True
            ),
            x="Frequency",
            y="Keyword",
            orientation="h",
            title="Most Frequently Used Keywords"
        ),
        use_container_width=True
    )

else:

    st.info("No keywords available for analysis.")


# -----------------------------------
# BUSINESS RECOMMENDATIONS
# -----------------------------------

st.divider()

st.header("Actionable Business Insights")


# -----------------------------------
# TOP RECOMMENDED BUSINESS ACTION
# -----------------------------------

st.subheader("Top Recommended Business Action")

st.warning(
    insights["recommended_action"]
)


# -----------------------------------
# CATEGORY-WISE RECOMMENDATIONS
# -----------------------------------

st.subheader("Recommended Business Actions")

st.caption(
    "The following actions are generated from category-wise "
    "negative feedback and assigned priority levels."
)

category_recommendations = insights.get(
    "category_recommendations",
    []
)

if category_recommendations:

    for item in category_recommendations:

        priority = item["priority"]
        category = item["category"]

        with st.expander(
            f"{priority} Priority: {category}"
        ):

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Negative Reviews",
                    item["negative_reviews"]
                )

            with col2:
                st.metric(
                    "Negative Feedback Rate",
                    f"{item['negative_percentage']}%"
                )

            st.markdown("**Recommended Action**")

            st.write(
                item["recommendation"]
            )

else:

    st.success(
        "No negative feedback detected. "
        "Continue monitoring customer satisfaction."
    )


# -----------------------------------
# KEY FINDINGS
# -----------------------------------

st.markdown("### Key Findings")

st.write(
    f"- Most discussed area: "
    f"{insights['most_discussed_category']}"
)

st.write(
    f"- Most concerning area by negative review volume: "
    f"{insights['most_concerning_category']}"
)

st.write(
    f"- Total customer feedback analysed: {total}"
)


# -----------------------------------
# FEEDBACK EXPLORER
# -----------------------------------

st.divider()

st.header("Customer Feedback Explorer")

sentiment_options = [
    "All",
    "Positive",
    "Negative",
    "Neutral"
]

selected_sentiment = st.selectbox(
    "Filter by sentiment",
    sentiment_options
)

explorer_df = df.copy()

if selected_sentiment != "All":

    explorer_df = explorer_df[
        explorer_df["sentiment"] == selected_sentiment
    ]

st.dataframe(
    explorer_df,
    use_container_width=True,
    hide_index=True
)


# -----------------------------------
# DOWNLOAD RESULTS
# -----------------------------------

st.divider()

st.header("Export Analysis")

csv_data = df.copy()

# Convert category lists to readable strings

csv_data["categories"] = csv_data[
    "categories"
].apply(lambda x: ", ".join(x))

download_data = csv_data.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Analysed Feedback CSV",
    data=download_data,
    file_name="customer_feedback_analysis.csv",
    mime="text/csv"
)


# -----------------------------------
# FOOTER
# -----------------------------------

st.success(
    "Customer feedback analysis completed successfully!"
)

st.caption(
    "Customer Feedback Intelligence | NLP-powered Analytics"
)
