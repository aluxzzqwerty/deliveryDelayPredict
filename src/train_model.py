import pandas as pd
import json
import os

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from joblib import dump

from xgboost import XGBClassifier

# Load Titanic dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Features & target
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = df["Survived"]

# Compute basic training stats for inference validation
num_features = ["Age", "SibSp", "Parch", "Fare"]
cat_features = ["Pclass", "Sex", "Embarked"]

stats = {}
for col in num_features:
    s = X[col].dropna()
    stats[col] = {
        "min": float(s.min()),
        "max": float(s.max()),
        "p01": float(s.quantile(0.01)),
        "p99": float(s.quantile(0.99)),
    }

categories = {
    "Sex": sorted(X["Sex"].dropna().str.lower().unique().tolist()),
    "Embarked": sorted(X["Embarked"].dropna().str.upper().unique().tolist()),
}

# Impute/scale/encode
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        # treat Pclass as categorical
        ("cat", cat_transformer, ["Pclass", "Sex", "Embarked"])
    ]
)

# Split (stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Base model
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

# Full pipeline
pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb)
])

# Parameter grid (tweak as needed)
param_grid = {
    "classifier__n_estimators": [200, 400],
    "classifier__max_depth": [3, 5],
    "classifier__learning_rate": [0.05, 0.1, 0.2],
    "classifier__reg_lambda": [1.0, 2.0]
}

# 5-fold stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# Fit
grid.fit(X_train, y_train)

# Best model & performance
best_model = grid.best_estimator_
print("Best CV score (accuracy):", grid.best_score_)
print("Best params:", grid.best_params_)

# Test accuracy
y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print("Test accuracy:", test_acc)

# Save artifacts
os.makedirs("models", exist_ok=True)

# Save best estimator (pipeline with preprocessor + tuned XGB)
dump(best_model, "models/titanic_xgb_best.joblib")

# Save metadata (bounds/categories)
with open("models/metadata.json", "w") as f:
    json.dump(
        {
            "bounds": stats,
            "categories": categories,
            "cv": {"n_splits": 5, "scoring": "accuracy"},
            "best_params": grid.best_params_,
            "best_cv_score": grid.best_score_,
            "test_accuracy": test_acc
        },
        f,
        indent=2
    )

# Save test set as CSV (features + target)
test_df = X_test.copy()
test_df["Survived"] = y_test.values
test_df.to_csv("models/test_set.csv", index=False)

print("Saved best model to models/titanic_xgb_best.joblib")
print("Saved metadata to models/metadata.json")
print("Saved test set to models/test_set.csv")
