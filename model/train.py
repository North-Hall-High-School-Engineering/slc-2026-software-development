import glob
import os
import shutil
import urllib.request
import zipfile

import evaluate
import numpy as np
import torch
from dataset_util import *
from datasets import Audio, Dataset
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    Trainer,
    TrainingArguments,
    WavLMForSequenceClassification,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]

    return {"accuracy": acc}


def main():
    base_dir = "data"

    ravdess_dir = download_ravdess(base_dir)
    tess_dir = download_tess(base_dir)
    cremad_dir = download_cremad(base_dir)
    emodb_dir = download_emodb(base_dir)

    dataset, emotion2id = load_all_datasets(
        ravdess_dir=ravdess_dir,
        tess_dir=tess_dir,
        cremad_dir=cremad_dir,
        emodb_dir=emodb_dir,
    )

    print(len(dataset))

    split = dataset.train_test_split(test_size=0.2, seed=67)
    dataset = {"train": split["train"], "test": split["test"]}

    val_split = dataset["train"].train_test_split(test_size=0.1, seed=67)
    dataset["train"] = val_split["train"]
    dataset["validation"] = val_split["test"]


if __name__ == "__main__":
    main()
