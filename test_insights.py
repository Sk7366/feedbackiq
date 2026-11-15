
import pandas as pd
from src.insights import generate_business_insights

# Sample analysed customer feedback
data = {
    "sentiment": [
        "Negative", "Negative", "Negative",
        "Negative", "Positive", "Positive",
        "Negative", "Neutral", "Positive",
        "Negative"
    ],

    "categories": [
        ["Product Quality"],
        ["Product Quality"],
        ["Delivery"],
        ["Delivery"],
        ["Delivery"],
        ["Customer Service"],
        ["Customer Service"],
        ["Pricing"],
        ["Pricing"],
        ["Product Quality"]
    ],

    "clean_feedback": [
        "damaged product quality",
        "defective product quality",
        "late delivery delayed shipping",
        "late delivery poor shipping",
        "fast delivery arrived quickly",
        "excellent customer support",
        "poor customer service response",
        "reasonable affordable pricing",
        "great price affordable",
        "faulty product quality"
    ]
}

df = pd.DataFrame(data)

insights = generate_business_insights(df)

print("\n===== BUSINESS INSIGHTS TEST =====")

print("\nTotal feedback:", insights["total_feedback"])

print("\nSentiment distribution:")
print(insights["sentiment_percentages"])

print("\nCategory analysis:")

for category, metrics in insights["category_analysis"].items():
    print(f"\n{category}")
    print(metrics)

print("\nMost discussed category:")
print(insights["most_discussed_category"])

print("\nMost concerning category:")
print(insights["most_concerning_category"])

print("\nRecommended action:")
print(insights["recommended_action"])

print("\nTop keywords:")
print(insights["top_keywords"])

print("\n===== TEST COMPLETED =====")
