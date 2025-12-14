from functools import partial

import torch
import torch.nn as nn
from datasets import Audio, load_dataset
from transformers import (
    HubertModel,
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
)

from models.combined import CombinedModel
from models.regressor import RegressorHead


def collate_fn_train(batch, feature_extractor):
    audio_arrays = [item["audio"]["array"] for item in batch]

    features = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    labels = torch.tensor(
        [
            [
                item["frustrated"],
                item["angry"],
                item["sad"],
                item["disgust"],
                item["excited"],
                item["fear"],
                item["neutral"],
                item["surprise"],
                item["happy"],
            ]
            for item in batch
        ],
        dtype=torch.float32,
    )

    return {
        "input_values": features["input_values"],
        "attention_mask": features["attention_mask"],
        "labels": labels,
    }


if __name__ == "__main__":
    dataset = load_dataset("AbstractTTS/IEMOCAP")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/hubert-base-ls960"
    )

    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    for param in hubert.parameters():
        param.requires_grad = False

    model = CombinedModel(hubert)

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_steps=10,
        logging_strategy="steps",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=partial(collate_fn_train, feature_extractor=feature_extractor),
    )

    trainer.train()
