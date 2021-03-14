import json
import random
from functools import partial
from pathlib import Path
from typing import Union, Dict

import librosa
import pandas as pd
import numpy as np
import torchaudio
from datasets import load_dataset, load_metric, DatasetDict, Dataset, Metric
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Trainer,
)
from transformers import TrainingArguments

import typer
import re

from data_collator import DataCollatorCTCWithPadding

chars_to_delete_regex = "[^ (A-z)\-\\d'éàùâ]"


def compute_metrics(pred, processor: Wav2Vec2Processor, wer_metric: Metric):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def prepare_dataset(batch, processor: Wav2Vec2Processor):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(
        batch["speech"], sampling_rate=batch["sampling_rate"][0]
    ).input_values

    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch


def resample(batch):
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 48_000, 16_000)
    batch["sampling_rate"] = 16_000
    return batch


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["text"]
    return batch


def extract_all_chars(batch: Union[DatasetDict, Dataset]) -> Dict:
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def remove_special_characters(
    batch: Union[DatasetDict, Dataset]
) -> Union[DatasetDict, Dataset]:
    batch["text"] = re.sub(chars_to_delete_regex, "", batch["sentence"]).lower() + " "
    return batch


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(
        dataset
    ), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    print(df)


def prepare_dataset(
    data_dir: Path = Path(__file__).parent / "data/cv-corpus/",
    output_path: Path = Path(__file__).parent / "data/vocab.json",
):
    french_common_voice_train = load_dataset(
        "common_voice", "fr", data_dir=str(data_dir), split="train"
    )[:10]

    load_dataset
    french_common_voice_test = load_dataset(
        "common_voice", "fr", data_dir=str(data_dir), split="test"
    )[:10]
    french_common_voice_val = load_dataset(
        "common_voice", "fr", data_dir=str(data_dir), split="validation"
    )[:10]

    french_common_voice_train = french_common_voice_train.remove_columns(
        [
            "accent",
            "age",
            "client_id",
            "down_votes",
            "gender",
            "locale",
            "segment",
            "up_votes",
        ]
    )
    french_common_voice_test = french_common_voice_test.remove_columns(
        [
            "accent",
            "age",
            "client_id",
            "down_votes",
            "gender",
            "locale",
            "segment",
            "up_votes",
        ]
    )
    french_common_voice_val = french_common_voice_val.remove_columns(
        [
            "accent",
            "age",
            "client_id",
            "down_votes",
            "gender",
            "locale",
            "segment",
            "up_votes",
        ]
    )

    typer.echo("Removing special characters")

    french_common_voice_train = french_common_voice_train.map(
        remove_special_characters, remove_columns=["sentence"]
    )
    french_common_voice_test = french_common_voice_test.map(
        remove_special_characters, remove_columns=["sentence"]
    )
    french_common_voice_val = french_common_voice_val.map(
        remove_special_characters, remove_columns=["sentence"]
    )

    show_random_elements(french_common_voice_train.remove_columns(["path"]))

    typer.echo("Extracting all characters")

    vocab_train = french_common_voice_train.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=french_common_voice_train.column_names,
    )
    vocab_test = french_common_voice_train.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=french_common_voice_test.column_names,
    )

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    print(vocab_dict)

    with open(output_path, "w+") as output_file:
        json.dump(vocab_dict, output_file)

    tokenizer = Wav2Vec2CTCTokenizer(
        output_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    typer.echo("Converting speech files to array")

    french_common_voice_train = french_common_voice_train.map(
        speech_file_to_array_fn, remove_columns=french_common_voice_train.column_names
    )
    french_common_voice_test = french_common_voice_test.map(
        speech_file_to_array_fn, remove_columns=french_common_voice_test.column_names
    )
    french_common_voice_val = french_common_voice_val.map(
        speech_file_to_array_fn, remove_columns=french_common_voice_val.column_names
    )

    typer.echo("Resampling dataset")

    french_common_voice_train = french_common_voice_train.map(resample, num_proc=4)
    french_common_voice_test = french_common_voice_test.map(resample, num_proc=4)
    french_common_voice_val = french_common_voice_val.map(resample, num_proc=4)

    rand_int = random.randint(0, len(french_common_voice_train))

    print("Target text:", french_common_voice_train[rand_int]["target_text"])
    print(
        "Input array shape:",
        np.asarray(french_common_voice_train[rand_int]["speech"]).shape,
    )
    print("Sampling rate:", french_common_voice_train[rand_int]["sampling_rate"])

    typer.echo("Preparing dataset")

    french_common_voice_train = french_common_voice_train.map(
        partial(prepare_dataset, processor=processor),
        remove_columns=french_common_voice_train.column_names,
        batch_size=8,
        num_proc=4,
        batched=True,
    )
    french_common_voice_test = french_common_voice_test.map(
        partial(prepare_dataset, processor=processor),
        remove_columns=french_common_voice_test.column_names,
        batch_size=8,
        num_proc=4,
        batched=True,
    )

    typer.echo("Initializing data collator")

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    wer_metric = load_metric("wer")

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        gradient_checkpointing=True,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    model.freeze_feature_extractor()

    training_args = TrainingArguments(
        output_dir="output/wav2vec-trained-french",
        group_by_length=True,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        save_steps=400,
        eval_steps=400,
        logging_steps=400,
        learning_rate=3e-4,
        warmup_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=french_common_voice_train,
        eval_dataset=french_common_voice_test,
        tokenizer=processor.feature_extractor,
    )


if __name__ == "__main__":
    typer.run(prepare_dataset)
