import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    return df


def preprocess_data(df: pd.DataFrame):
    target = "Late_delivery_risk"

    cols_to_drop = [
    # target
    'Late_delivery_risk',

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

    # geo
    'Latitude', 'Longitude'
]

    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Разделяем X и y
    X = df.drop(columns=[target])
    y = df[target]

    # Кодируем категориальные признаки
    categorical_cols = X.select_dtypes(include=["object"]).columns

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Делим на train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
