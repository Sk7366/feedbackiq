
import pandas as pd
import re


def clean_text(text):
    """
    Basic text cleaning for customer feedback.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove unnecessary leading/trailing spaces
    text = text.strip()

    # Replace multiple spaces and newlines with one space
    text = re.sub(r"\s+", " ", text)

    return text


def preprocess_feedback(df, feedback_column="feedback"):
    """
    Validate and preprocess customer feedback data.
    """

    if feedback_column not in df.columns:
        raise ValueError(
            f"Required column '{feedback_column}' not found."
        )

    # Work on a copy to preserve original data
    df = df.copy()

    # Remove rows with missing feedback
    df = df.dropna(subset=[feedback_column])

    # Clean feedback
    df["clean_feedback"] = df[feedback_column].apply(clean_text)

    # Remove empty feedback
    df = df[df["clean_feedback"] != ""]

    # Reset index
    df = df.reset_index(drop=True)

    return df
