from pydantic import BaseModel

class ModelConfig(BaseModel):
    model_name: str = "tag_transformer"

    embed_dim: int = 256
    num_layers: int = 8
    num_heads: int = 8
    dim_feedforward: int = 256 * 4

class TrainConfig(BaseModel):
    experiment_name: str = "20250509b"
    device: str = "cuda"
    seed: int = 114514
    mask_prob: float = 0.10
    batch_size: int = 768
    learning_rate: float = 1e-4
    epochs: int = 200
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
Epoch 200 Train Loss: 0.6180
Epoch 200 Val Loss: 0.7922, Top 1 Acc: 0.8718, Top 5 Acc: 0.8997
"""