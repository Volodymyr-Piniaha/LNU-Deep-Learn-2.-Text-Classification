import pandas as pd
import joblib

def main():

    test_df = pd.read_csv("test.csv")

    pipeline = joblib.load("model.pkl")

    predictions = pipeline.predict(test_df["text"])

    submission = pd.DataFrame({
        "id": test_df["id"],
        "label": predictions
    })

    submission.to_csv("submission.csv", index=False)
    print("Predictions saved to submission.csv")

if __name__ == "__main__":
    main()