import joblib
from sklearn.model_selection import train_test_split

from data_preprocessing import load_data, clean_data, split_features_target
from model import build_preprocessor, build_pipeline

# Load & clean
df = load_data("DataCoSupplyChainDataset.csv")
df = clean_data(df)

X, y, num_features, cat_features = split_features_target(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = build_preprocessor(num_features, cat_features)
pipe = build_pipeline(preprocessor)

pipe.fit(X_train, y_train)

# Save model
joblib.dump(pipe, "artifacts/model.joblib")

print("Model saved to artifacts/model.joblib")
