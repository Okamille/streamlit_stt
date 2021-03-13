from pathlib import Path

from datasets import load_dataset, load_metric
import typer


def train_model(data_dir: Path = Path(__file__).parent / "files/dataset/"):
    french_common_voice_train = load_dataset("common_voice", "tr", data_dir=data_dir, split="train")
    french_common_voice_test = load_dataset("common_voice", "tr", data_dir=data_dir, split="test")
    french_common_voice_val = load_dataset("common_voice", "tr", data_dir=data_dir, split="validation")

if __name__ == "__main__":
    typer.run(train_model)