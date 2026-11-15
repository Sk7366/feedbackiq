
import pandas as pd
from pathlib import Path

file_path = Path("customer_feedback_analysis.csv")

if not file_path.exists():
    raise FileNotFoundError(
        "Download the analysed CSV and place it in the project folder."
    )

df = pd.read_csv(file_path)

required_columns = [
    "feedback",
    "clean_feedback",
    "positive_score",
    "negative_score",
    "neutral_score",
    "compound_score",
    "sentiment",
    "categories"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    print("Missing columns:", missing)
else:
    print("All required columns are present.")

    print("\nTotal feedback records:", len(df))

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    print("\nCompound score range:")
    print(df["compound_score"].min(),
          "to",
          df["compound_score"].max())

    valid_sentiments = {"Positive", "Negative", "Neutral"}

    invalid = set(df["sentiment"].dropna()) - valid_sentiments

    if invalid:
        print("\nInvalid sentiment labels:", invalid)
    else:
        print("\nSentiment labels are valid.")

    score_columns = [
        "positive_score",
        "negative_score",
        "neutral_score",
        "compound_score"
    ]

    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_scores = (
        df[score_columns].isna().any().any()
        or not df["compound_score"].between(-1, 1).all()
        or not df[
            ["positive_score", "negative_score", "neutral_score"]
        ].apply(lambda col: col.between(0, 1).all()).all()
    )

    print(
        "\nScore validation:",
        "FAILED" if invalid_scores else "PASSED"
    )

    print("\nSample analysed feedback:")
    print(
        df[[
            "feedback",
            "compound_score",
            "sentiment",
            "categories"
        ]].head(5).to_string(index=False)
    )
