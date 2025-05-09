from config.configs import ExperimentConfig
from typing import Dict
from utils.seeds import seed_all
from testing.tag_transformer_test import *
from utils.tag_transformer_tsne import *

def main(steps: Dict):
    config = ExperimentConfig()
    seed_all(config.seed)
    
    if steps.get("test_tag_transformer_mask_predict", False):
        test_tag_transformer_mask_predict()
    if steps.get("test_tag_transformer_occlusion", False):
        test_tag_transformer_occlusion()
    if steps.get("test_tag_transformer_tsne", False):
        test_tag_transformer_tsne()


if __name__ == "__main__":
    main(steps={
        "test_tag_transformer_mask_predict": False,
        "test_tag_transformer_occlusion": False,
        "test_tag_transformer_tsne": True,
    })