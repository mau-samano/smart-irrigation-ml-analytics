import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Base path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "processed", "clean_data_v1.csv")


def load_data():
    return pd.read_csv(data_path)


def preprocess(df):
    df_model = df.drop(["id"], axis=1)

    df_encoded = df_model.copy()
    le = LabelEncoder()

    for col in df_encoded.select_dtypes(include="object").columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    X = df_encoded.drop("Irrigation_Need", axis=1)
    y = df_encoded["Irrigation_Need"]

    return X, y


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    print(f"\nROC AUC Score: {auc:.4f}")


def main():
    df = load_data()

    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
