from typing import Dict
from training.train_tag_transformer import *


def main(steps: Dict):
    if steps.get("train_tag_transformer", False):
        train_tag_transformer()


if __name__ == "__main__":
    main(steps={
        "train_tag_transformer": True,
    })