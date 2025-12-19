import pandas as pd
from typing import Tuple, List

TARGET = "Late_delivery_risk"

COLS_TO_DROP = [
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
    'Product Status',

    # geo
    'Latitude', 'Longitude'
]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin1")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])


def split_features_target(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Returns X, y, numeric_features, categorical_features
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    return X, y, num_features, cat_features
