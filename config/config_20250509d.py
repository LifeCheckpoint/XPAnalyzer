from pydantic import BaseModel

class BaseInfoConfig(BaseModel):
    # 基本配置参数
    experiment_name: str = "20250509d"
    seed: int = 114514
    device: str = "cuda"

class ModelConfig(BaseInfoConfig):
    # 模型配置参数
    model_name: str = "tag_transformer"
    embed_dim: int = 256
    num_layers: int = 8
    num_heads: int = 8
    dim_feedforward: int = 256 * 4

class DatasetConfig(BaseInfoConfig):
    # 数据集配置参数
    characters_tags_dataset_path: str = "E:/Dataset/bangumi/data_min.json"

class TrainConfig(BaseInfoConfig):
    # 训练配置参数
    mask_prob: float = 0.10
    batch_size: int = 768
    learning_rate: float = 8e-5
    epochs: int = 150
    save_interval: int = 30
    save_dir: str = f"./saves/{BaseInfoConfig().experiment_name}"

    eval_size: float = 0.1

class EvaluateConfig(BaseInfoConfig):
    # 评估配置参数
    batch_size: int = 768
    top_k: int = 10

class TestConfig(BaseInfoConfig):
    # 测试配置参数
    batch_size: int = 512
    top_k: int = 10
    model_tag_transformer_path: str = f"./saves/{BaseInfoConfig().experiment_name}/{ModelConfig().model_name}_epoch_120.pth"
    tsne_visual_num_words: int = 100
    tsne_n_iter: int = 500
    tsne_perplexity: int = 20
    tsne_image_output_dir: str = f"./saves/{BaseInfoConfig().experiment_name}"

class ExperimentConfig(BaseInfoConfig):
    # 实验配置参数
    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = DatasetConfig()
    train: TrainConfig = TrainConfig()
    evaluate: EvaluateConfig = EvaluateConfig()
    test: TestConfig = TestConfig()

"""

"""