from pydantic import BaseModel

class ModelConfig(BaseModel):
    model_name: str = "tag_transformer"

    embed_dim: int = 128
    num_layers: int = 10
    num_heads: int = 8
    dim_feedforward: int = 128 * 6

class TrainConfig(BaseModel):
    experiment_name: str = "20250509a"
    device: str = "cuda"
    seed: int = 114514
    mask_prob: float = 0.12
    batch_size: int = 768
    learning_rate: float = 1e-4
    epochs: int = 500
    save_interval: int = 50
    save_path: str = "./saves"

    characters_tags_dataset_path: str = "E:/Dataset/bangumi/data_min.json"
    eval_size: float = 0.1

class EvaluateConfig(BaseModel):
    device: str = "cuda"
    seed: int = 114514
    batch_size: int = 768
    top_k: int = 10

class TestConfig(BaseModel):
    device: str = "cuda"
    seed: int = 114514
    batch_size: int = 768
    top_k: int = 10
    model_tag_transformer_path: str = f"./saves/{TrainConfig().experiment_name}/{ModelConfig().model_name}_epoch_150.pth"

class BaseConfig(BaseModel):
    model_config: ModelConfig = ModelConfig()
    train_config: TrainConfig = TrainConfig()
    test_config: EvaluateConfig = EvaluateConfig()

"""
Epoch 500 Train Loss: 0.6930
Epoch 500 Val Loss: 0.8551, Top 1 Acc: 0.8630, Top 5 Acc: 0.8919
"""