import os
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk

def load_data(path, text_col="sentences", label_col="labels"):
    data = pd.read_json(path, lines=True)
    data = data[[text_col, label_col, 'project_id']]
    data = data.rename(columns={text_col: "sentences", label_col: "labels"})
    return Dataset.from_pandas(data[["sentences", "labels", "project_id"]])

def get_dataset(cache_dir="cache/dataset_cache", text_col="sentences", label_col="labels"):
    # If cache exists, load directly
    if cache_dir is not None and os.path.exists(cache_dir):
        print(f"Loading dataset from cache: {cache_dir}")
        dataset = load_from_disk(cache_dir)
        return dataset

    # Otherwise, build the dataset
    train_path = "data/legitimacy_train_.jsonl"
    val_path = "data/legitimacy_val_.jsonl"
    test_path = "data/legitimacy_test_.jsonl"

    train_data = load_data(train_path, text_col, label_col)
    val_data = load_data(val_path, text_col, label_col)
    test_data = load_data(test_path, text_col, label_col)

    dataset = DatasetDict({
        "train": train_data,
        "val": val_data,
        "test": test_data
    })

    # Save to cache for future use
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        dataset.save_to_disk(cache_dir)
        print(f"Dataset saved to cache: {cache_dir}")

    return dataset
