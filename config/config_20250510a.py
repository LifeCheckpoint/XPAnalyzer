from pydantic import BaseModel

class BaseInfoConfig(BaseModel):
    # 基本配置参数
    experiment_name: str = "20250510a"
    seed: int = 114514
    device: str = "cuda"
    eps: float = 0.1

class ModelConfig(BaseInfoConfig):
    # 模型配置参数
    model_name: str = "tag_transformer"
    embed_dim: int = 128
    num_layers: int = 8
    num_heads: int = 8
    dim_feedforward: int = 128 * 4

class DatasetConfig(BaseInfoConfig):
    # 数据集配置参数
    characters_tags_dataset_path: str = "./datasets/data_min.json"
    tag_freq_threshold: int = 3
    tags_weights_dataset_path: str = "./datasets/tag_weights.json"
    moe_ys_al_state_path: str = "./saves/state.json"
    embedding_saving_path: str = f"./saves/{BaseInfoConfig().experiment_name}/tag_embedding_dict.pt"
    tags_embedding_distance_pickle_path: str = f"./saves/{BaseInfoConfig().experiment_name}/tag_embedding_distance.pkl"

class TrainConfig(BaseInfoConfig):
    # 训练配置参数
    batch_size: int = 512
    learning_rate: float = 1e-4
    epochs: int = 150
    save_interval: int = 50
    save_dir: str = f"./saves/{BaseInfoConfig().experiment_name}"
    eval_size: float = 0.1

    mask_prob: float = 0.10
    contrastive_temperature: float = 0.15
    mask_predict_loss_weight: float = 1
    contrastive_loss_weight: float = 0.2

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

class MarkerConfig(BaseInfoConfig):
    # 打分配置参数
    distance_transform_scale: float = 0.5
    distance_transform_power: float = 0.2 # 越低越拉开差距，越高越容易共同带上相关标签
    pre_weights_transform_power: float = 0 # 越低越接近原始值，越高相对差距越大

class ExperimentConfig(BaseInfoConfig):
    # 实验配置参数
    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = DatasetConfig()
    train: TrainConfig = TrainConfig()
    evaluate: EvaluateConfig = EvaluateConfig()
    test: TestConfig = TestConfig()
    mark: MarkerConfig = MarkerConfig()

"""
Epoch 150 Train Loss: Critetion = 0.7114, Contrastive = 0.1167
Epoch 150 Val Critetion Loss: 0.7348, Top 1 Acc: 0.8777, Top 5 Acc: 0.9034
"""