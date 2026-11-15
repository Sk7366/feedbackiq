
import re
from collections import Counter


# ==========================================
# 1. BUSINESS FEEDBACK CATEGORIES
# ==========================================

CATEGORY_KEYWORDS = {
    "Product Quality": [
        "quality", "damaged", "defective",
        "broken", "faulty", "product"
    ],
    "Delivery": [
        "delivery", "shipping", "shipped",
        "late", "delayed", "arrived"
    ],
    "Customer Service": [
        "support", "service", "response",
        "responded", "staff", "team"
    ],
    "Pricing": [
        "price", "expensive", "costly",
        "pricing", "affordable", "cheap"
    ],
    "Refunds": [
        "refund", "return", "reimbursement",
        "money back"
    ],
    "Website": [
        "website", "checkout", "interface",
        "application", "app", "navigation"
    ],
    "Packaging": [
        "packaging", "package", "box",
        "wrapped"
    ]
}


# ==========================================
# 2. STOPWORDS
# ==========================================

STOPWORDS = {
    "the", "is", "was", "and", "a", "an",
    "to", "of", "for", "in", "it", "this",
    "that", "my", "with", "but", "very",
    "as", "on", "i", "we", "our", "be",
    "has", "have", "had", "are", "were",
    "not", "will", "could", "would", "been",
    "you", "your", "they", "their", "from",
    "too", "can", "all", "also", "its"
}


# ==========================================
# 3. FEEDBACK CATEGORIZATION
# ==========================================

def categorize_feedback(text):
    """Identify business categories in customer feedback."""

    text = str(text).lower()
    matched_categories = []

    for category, keywords in CATEGORY_KEYWORDS.items():

        for keyword in keywords:

            pattern = r"\b" + re.escape(keyword) + r"\b"

            if re.search(pattern, text):
                matched_categories.append(category)
                break

    return matched_categories if matched_categories else ["Other"]


def add_categories(df, text_column="clean_feedback"):
    """Add category information to the dataset."""

    df = df.copy()

    df["categories"] = df[text_column].apply(
        categorize_feedback
    )

    return df


# ==========================================
# 4. KEYWORD EXTRACTION
# ==========================================

def extract_keywords(text, top_n=5):
    """Extract frequently occurring meaningful words."""

    words = re.findall(
        r"\b[a-zA-Z]{3,}\b",
        str(text).lower()
    )

    meaningful_words = [
        word for word in words
        if word not in STOPWORDS
    ]

    return Counter(meaningful_words).most_common(top_n)


def get_top_keywords(
    df,
    text_column="clean_feedback",
    top_n=10
):
    """Extract frequent keywords across all feedback."""

    all_words = []

    for text in df[text_column]:

        words = re.findall(
            r"\b[a-zA-Z]{3,}\b",
            str(text).lower()
        )

        all_words.extend(
            word for word in words
            if word not in STOPWORDS
        )

    return Counter(all_words).most_common(top_n)


# ==========================================
# 5. BUSINESS INSIGHTS GENERATION
# ==========================================

def generate_business_insights(df):
    """Generate business insights from analysed feedback."""

    if df.empty:
        raise ValueError(
            "No feedback available for analysis."
        )

    required_columns = {
        "sentiment",
        "categories"
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Run sentiment analysis and categorisation first."
        )

    total_feedback = len(df)

    # --------------------------------------
    # 5.1 OVERALL SENTIMENT DISTRIBUTION
    # --------------------------------------

    sentiment_counts = (
        df["sentiment"].value_counts().to_dict()
    )

    positive = sentiment_counts.get("Positive", 0)
    negative = sentiment_counts.get("Negative", 0)
    neutral = sentiment_counts.get("Neutral", 0)

    sentiment_percentages = {
        "Positive": round(
            positive / total_feedback * 100, 2
        ),
        "Negative": round(
            negative / total_feedback * 100, 2
        ),
        "Neutral": round(
            neutral / total_feedback * 100, 2
        )
    }

    # --------------------------------------
    # 5.2 CATEGORY-WISE ANALYSIS
    # --------------------------------------

    category_data = df.explode("categories").copy()

    category_data = category_data.dropna(
        subset=["categories"]
    )

    category_counts = (
        category_data["categories"].value_counts().to_dict()
    )

    negative_data = category_data[
        category_data["sentiment"] == "Negative"
    ]

    negative_category_counts = (
        negative_data["categories"].value_counts().to_dict()
    )

    # --------------------------------------
    # 5.3 CATEGORY METRICS AND PRIORITIES
    # --------------------------------------

    category_analysis = {}

    for category, count in category_counts.items():

        negative_count = negative_category_counts.get(
            category, 0
        )

        negative_rate = round(
            negative_count / count * 100, 2
        )

        if negative_count >= 10 and negative_rate >= 30:
            priority = "High"

        elif negative_count >= 4 or negative_rate >= 40:
            priority = "Medium"

        else:
            priority = "Low"

        category_analysis[category] = {
            "total_reviews": count,
            "negative_reviews": negative_count,
            "negative_percentage": negative_rate,
            "priority": priority
        }

    # --------------------------------------
    # 5.4 MOST DISCUSSED CATEGORY
    # --------------------------------------

    most_discussed = (
        max(category_counts, key=category_counts.get)
        if category_counts
        else "None"
    )

    # --------------------------------------
    # 5.5 MOST CONCERNING CATEGORY
    # --------------------------------------

    eligible_categories = {
        category: metrics
        for category, metrics in category_analysis.items()
        if metrics["total_reviews"] >= 3
    }

    if eligible_categories:

        most_concerning = max(
            eligible_categories,
            key=lambda category: (
                eligible_categories[category]["negative_reviews"],
                eligible_categories[category]["negative_percentage"]
            )
        )

    else:
        most_concerning = "None"

    # --------------------------------------
    # 5.6 KEYWORD EXTRACTION
    # --------------------------------------

    top_keywords = get_top_keywords(df)

    # --------------------------------------
    # 5.7 CATEGORY-SPECIFIC RECOMMENDATIONS
    # --------------------------------------

    recommendations = {

        "Product Quality":
            "Investigate recurring product defects and introduce stricter quality checks.",

        "Delivery":
            "Review logistics partners, investigate delayed shipments and improve delivery tracking.",

        "Customer Service":
            "Improve response times and provide additional customer support training.",

        "Pricing":
            "Review pricing strategy and evaluate customer perceived value.",

        "Refunds":
            "Simplify refund procedures and reduce resolution time.",

        "Website":
            "Improve website navigation, checkout usability and application performance.",

        "Packaging":
            "Improve packaging standards and introduce better product protection.",

        "Other":
            "Review customer comments to identify emerging customer concerns."
    }

    category_recommendations = []

    for category, metrics in category_analysis.items():

        # Skip categories without negative feedback
        if metrics["negative_reviews"] == 0:
            continue

        category_recommendations.append({

            "category": category,

            "priority": metrics["priority"],

            "negative_reviews": metrics["negative_reviews"],

            "negative_percentage": metrics["negative_percentage"],

            "recommendation": recommendations.get(
                category,
                recommendations["Other"]
            )
        })

    # --------------------------------------
    # 5.8 SORT RECOMMENDATIONS BY PRIORITY
    # --------------------------------------

    priority_order = {
        "High": 1,
        "Medium": 2,
        "Low": 3
    }

    category_recommendations.sort(
        key=lambda item: (
            priority_order[item["priority"]],
            -item["negative_reviews"]
        )
    )

    # --------------------------------------
    # 5.9 SELECT TOP RECOMMENDED ACTION
    # --------------------------------------

    suggested_action = (

        category_recommendations[0]["recommendation"]

        if category_recommendations

        else "No negative feedback detected. Continue monitoring customer satisfaction."
    )

    # --------------------------------------
    # 5.10 RETURN ALL BUSINESS INSIGHTS
    # --------------------------------------

    return {

        "total_feedback": total_feedback,

        "sentiment_counts": sentiment_counts,

        "sentiment_percentages": sentiment_percentages,

        "category_counts": category_counts,

        "negative_category_counts": negative_category_counts,

        "category_negative_rates": {
            category: metrics["negative_percentage"]
            for category, metrics in category_analysis.items()
        },

        "category_analysis": category_analysis,

        "most_discussed_category": most_discussed,

        "most_concerning_category": most_concerning,

        "top_keywords": top_keywords,

        "recommended_action": suggested_action,

        "category_recommendations": category_recommendations
    }
