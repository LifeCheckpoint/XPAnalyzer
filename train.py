from config.configs import ExperimentConfig
from typing import Dict
from utils.seeds import seed_all
from training.train_tag_transformer import *

def main(steps: Dict):
    config = ExperimentConfig()
    seed_all(config.seed)
    
    if steps.get("train_tag_transformer", False):
        train_tag_transformer()


if __name__ == "__main__":
    main(steps={
        "train_tag_transformer": True,
    })