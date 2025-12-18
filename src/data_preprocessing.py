import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    return df


def preprocess_data(df: pd.DataFrame):
    target = "Late_delivery_risk"

    cols_to_drop = [
    # leakage
    'Delivery Status',
    'Order Status',
    'shipping date (DateOrders)',
    'Days for shipping (real)',

    # IDs
    'Order Id', 'Order Item Id', 'Order Item Cardprod Id',
    'Order Customer Id', 'Customer Id',
    'Product Card Id', 'Category Id',
    'Department Id', 'Product Category Id',

    # PII
    'Customer Fname', 'Customer Lname', 'Customer Email',
    'Customer Password', 'Customer Street',
    'Customer Zipcode', 'Order Zipcode',

    # text
    'Product Description', 'Product Image',
    'Product Name', 'Category Name', 'Department Name',

    # constant
    'Product Status'

    # geo
    'Latitude', 'Longitude'
]

    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Разделяем X и y
    X = df.drop(columns=[target])
    y = df[target]

    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # preprocessor = ColumnTransformer(
    #     transformers = [
    #         ("num", num_transformer, num_features),
    #         ("cat", cat_transformer, cat_features)
    #     ]
    # )

    # xgb = XGBClassifier(
    #     objective="binary:logistic",
    #     eval_metric="logloss",
    #     tree_method="hist",
    #     random_state=42,
    #     n_jobs=-1
    # )

    # Делим на train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
