
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd


# Initialise VADER
analyzer = SentimentIntensityAnalyzer()


def classify_sentiment(compound_score):
    """
    Classify feedback based on VADER compound score.
    """

    if compound_score >= 0.05:
        return "Positive"

    elif compound_score <= -0.05:
        return "Negative"

    else:
        return "Neutral"


def analyze_sentiment(text):
    """
    Analyse an individual customer feedback entry.
    """

    scores = analyzer.polarity_scores(str(text))

    compound = scores["compound"]

    sentiment = classify_sentiment(compound)

    return {
        "positive_score": scores["pos"],
        "negative_score": scores["neg"],
        "neutral_score": scores["neu"],
        "compound_score": compound,
        "sentiment": sentiment
    }


def analyze_dataset(df, text_column="clean_feedback"):
    """
    Analyse sentiment for every feedback entry.
    """

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in dataset."
        )

    df = df.copy()

    # Apply sentiment analysis to every feedback
    results = df[text_column].apply(analyze_sentiment)

    # Convert results into separate columns
    results_df = pd.DataFrame(results.tolist(), index=df.index)

    # Combine original data with sentiment results
    df = pd.concat([df, results_df], axis=1)

    return df
