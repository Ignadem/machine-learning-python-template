import json
from pathlib import Path

import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FOLDER = PROJECT_ROOT / "data" / "raw"
PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed"
MODELS_FOLDER = PROJECT_ROOT / "models"
AIRBNB_DATA_URL = "https://breathecode.herokuapp.com/asset/internal-link?id=927&path=AB_NYC_2019.csv"
RAW_CSV_PATH = RAW_FOLDER / "AB_NYC_2019.csv"


def create_folders():
    RAW_FOLDER.mkdir(parents=True, exist_ok=True)
    PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)


def step_1_data_collection():
    """Step 1: Download and load the Airbnb NYC dataset."""

    if RAW_CSV_PATH.exists():
        dataset = pd.read_csv(RAW_CSV_PATH)
        return dataset

    response = requests.get(AIRBNB_DATA_URL, timeout=120)
    response.raise_for_status()
    RAW_CSV_PATH.write_bytes(response.content)
    dataset = pd.read_csv(RAW_CSV_PATH)
    return dataset


def build_eda_summary(dataframe, title):
    return {
        "title": title,
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "duplicates": int(dataframe.duplicated().sum()),
        "missing_values": dataframe.isna().sum().to_dict(),
        "price_describe": dataframe["price"].describe().to_dict(),
        "room_type_distribution": dataframe["room_type"].value_counts().to_dict(),
        "neighbourhood_group_distribution": dataframe["neighbourhood_group"].value_counts().to_dict(),
    }


def step_2_exploration_and_cleaning(dataframe):
    """Step 2: Basic EDA and data cleaning."""

    raw_summary = build_eda_summary(dataframe, "raw_dataset")
    with open(MODELS_FOLDER / "eda_raw_summary.json", "w", encoding="utf-8") as file:
        json.dump(raw_summary, file, indent=2)

    columns_we_use = [
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "price",
    ]
    cleaned_dataframe = dataframe[columns_we_use].copy()
    cleaned_dataframe = cleaned_dataframe.drop_duplicates()
    cleaned_dataframe = cleaned_dataframe[cleaned_dataframe["price"] > 0]

    cleaned_summary = build_eda_summary(cleaned_dataframe, "cleaned_dataset")
    with open(MODELS_FOLDER / "eda_clean_summary.json", "w", encoding="utf-8") as file:
        json.dump(cleaned_summary, file, indent=2)

    cleaned_dataframe.to_csv(PROCESSED_FOLDER / "airbnb_clean.csv", index=False)
    return cleaned_dataframe


def build_preprocessing_pipeline():
    numeric_columns = [
        "latitude",
        "longitude",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]
    text_columns = ["neighbourhood_group", "neighbourhood", "room_type"]

    numeric_steps = Pipeline(
        steps=[
            ("fill_missing", SimpleImputer(strategy="median")),
            ("scale_values", StandardScaler()),
        ]
    )

    text_steps = Pipeline(
        steps=[
            ("fill_missing", SimpleImputer(strategy="most_frequent")),
            ("create_columns", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numbers", numeric_steps, numeric_columns),
            ("text", text_steps, text_columns),
        ]
    )


def step_3_feature_engineering(cleaned_dataframe):
    """Step 3: Split data and preprocess train/test."""

    target = cleaned_dataframe["price"]
    features = cleaned_dataframe.drop(columns=["price"])
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    preprocessing_pipeline = build_preprocessing_pipeline()
    x_train_ready = preprocessing_pipeline.fit_transform(x_train)
    x_test_ready = preprocessing_pipeline.transform(x_test)
    final_column_names = preprocessing_pipeline.get_feature_names_out()

    x_train_df = pd.DataFrame(x_train_ready, columns=final_column_names)
    x_test_df = pd.DataFrame(x_test_ready, columns=final_column_names)
    y_train_df = y_train.to_frame(name="price")
    y_test_df = y_test.to_frame(name="price")
    return x_train_df, x_test_df, y_train_df, y_test_df


def step_4_save_processed_data(x_train_df, x_test_df, y_train_df, y_test_df):
    """Step 4: Save processed train/test files."""

    x_train_df.to_csv(PROCESSED_FOLDER / "X_train.csv", index=False)
    x_test_df.to_csv(PROCESSED_FOLDER / "X_test.csv", index=False)
    y_train_df.to_csv(PROCESSED_FOLDER / "y_train.csv", index=False)
    y_test_df.to_csv(PROCESSED_FOLDER / "y_test.csv", index=False)
    clean_train = x_train_df.copy()
    clean_train["price"] = y_train_df["price"].values
    clean_test = x_test_df.copy()
    clean_test["price"] = y_test_df["price"].values
    clean_train.to_csv(PROCESSED_FOLDER / "clean_train.csv", index=False)
    clean_test.to_csv(PROCESSED_FOLDER / "clean_test.csv", index=False)


def main():
    print("Step 0: Create folders")
    create_folders()

    print("Step 1: Data collection")
    raw_dataframe = step_1_data_collection()

    print("Step 2: Exploration and data cleaning")
    cleaned_dataframe = step_2_exploration_and_cleaning(raw_dataframe)

    print("Step 3: Feature engineering (split + preprocessing)")
    x_train_df, x_test_df, y_train_df, y_test_df = step_3_feature_engineering(cleaned_dataframe)

    print("Step 4: Save processed data")
    step_4_save_processed_data(x_train_df, x_test_df, y_train_df, y_test_df)

    print("Pipeline completed.")
    print(f"Raw data path: {RAW_CSV_PATH}")
    print(f"Rows after cleaning: {len(cleaned_dataframe)}")
    print(f"Train shape: {x_train_df.shape}")
    print(f"Test shape: {x_test_df.shape}")
    print(f"EDA files: {MODELS_FOLDER / 'eda_raw_summary.json'} and {MODELS_FOLDER / 'eda_clean_summary.json'}")


if __name__ == "__main__":
    main()
