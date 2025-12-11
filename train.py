import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def main():
    train_df = pd.read_csv("train.csv")

    X_train, X_val, y_train, y_val = train_test_split(
        train_df["text"], train_df["label"], test_size=0.2, random_state=42, stratify=train_df["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),  # unigrams + bigrams
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs"))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    print("Validation results:\n", classification_report(y_val, y_pred))

    joblib.dump(pipeline, "model.pkl")
    print("✅ Model saved to model.pkl")



if __name__ == "__main__":
    main()