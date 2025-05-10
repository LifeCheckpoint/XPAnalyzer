from pydantic import BaseModel

class BaseInfoConfig(BaseModel):
    # 基本配置参数
    experiment_name: str = "20250510b"
    seed: int = 114514
    device: str = "cuda"

class ModelConfig(BaseInfoConfig):
    # 模型配置参数
    model_name: str = "tag_transformer"
    embed_dim: int = 64
    num_layers: int = 8
    num_heads: int = 8
    dim_feedforward: int = 128 * 3

class DatasetConfig(BaseInfoConfig):
    # 数据集配置参数
    characters_tags_dataset_path: str = "E:/Dataset/bangumi/data_min.json"
    tag_freq_threshold: int = 3

class TrainConfig(BaseInfoConfig):
    # 训练配置参数
    batch_size: int = 768
    learning_rate: float = 1e-4
    epochs: int = 150
    save_interval: int = 50
    save_dir: str = f"./saves/{BaseInfoConfig().experiment_name}"
    eval_size: float = 0.1

    mask_prob: float = 0.10
    contrastive_temperature: float = 0.15
    mask_predict_loss_weight: float = 1
    contrastive_loss_weight: float = 0.3

class EvaluateConfig(BaseInfoConfig):
    # 评估配置参数
    batch_size: int = 768
    top_k: int = 10

class TestConfig(BaseInfoConfig):
    # 测试配置参数
    batch_size: int = 512
    top_k: int = 10
    model_tag_transformer_path: str = f"./saves/{BaseInfoConfig().experiment_name}/{ModelConfig().model_name}_epoch_150.pth"
    tsne_visual_num_words: int = 100
    tsne_n_iter: int = 500
    tsne_perplexity: int = 20
    tsne_image_output_dir: str = f"./saves/{BaseInfoConfig().experiment_name}"
    emb_dist_output_dir: str = f"./saves/{BaseInfoConfig().experiment_name}"

class ExperimentConfig(BaseInfoConfig):
    # 实验配置参数
    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = DatasetConfig()
    train: TrainConfig = TrainConfig()
    evaluate: EvaluateConfig = EvaluateConfig()
    test: TestConfig = TestConfig()

"""
"""