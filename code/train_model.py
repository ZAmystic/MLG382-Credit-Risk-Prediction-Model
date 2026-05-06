import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ---------------- LOAD DATA ----------------
df = pd.read_csv("cleaned_german_credit_data.csv")

# ---------------- FEATURES / TARGET ----------------
X = df.drop("Risk", axis=1)
y = df["Risk"].map({"good": 1, "bad": 0})  # 1 = low risk, 0 = high risk

# ---------------- COLUMN TYPES ----------------
numeric_features = ["Age", "Job", "Credit amount", "Duration"]
categorical_features = [
    "Sex",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Purpose"
]

# ---------------- PREPROCESSING ----------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ---------------- MODEL PIPELINE ----------------
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

# ---------------- TRAIN / TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- TRAIN ----------------
model_pipeline.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
y_pred = model_pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------- SAVE MODEL ----------------
joblib.dump(model_pipeline, "credit_model.pkl")
print("\nSaved model as credit_model.pkl")