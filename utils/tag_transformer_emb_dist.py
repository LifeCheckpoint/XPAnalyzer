from config.configs import ExperimentConfig
from data.tag_dataset import collate_fn
from typing import List, Dict, FrozenSet, Tuple
import itertools
import torch

cfg = ExperimentConfig()

class TagTransformerEmbeddingDistance:
    """
    统计计算标签嵌入距离（包含标签与自身的距离）
    """
    def __init__(self, model, vocab, device=cfg.device):
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.device = device
        self.id_to_vocab = {idx: token for token, idx in vocab.items()}

    def compute_embedding_distance(self, tags: List[str]) -> Tuple[Dict[FrozenSet[str], float], Dict[str, torch.Tensor]]:
        """
        计算标签嵌入距离（包含标签与自身的距离）
        """
        # 将标签转换为索引
        tags_batch = self.prepare_batch([[tag] for tag in tags])
        padded_input = collate_fn(tags_batch, pad_token_id=self.vocab['[PAD]'])
        padded_input = {k: v.to(self.device) for k, v in padded_input.items()}

        # 获取标签嵌入
        with torch.no_grad():
            embeddings = self.model(padded_input["input_ids"], padded_input["attention_mask"])["encoded"]
            embeddings = torch.reshape(embeddings, (-1, cfg.model.embed_dim))

        # 计算所有标签对的距离（包括自身）
        return self.calculate_distances(tags, embeddings), {tag: embeddings[i] for i, tag in enumerate(tags)}
    
    def prepare_batch(self, input_tags_batch):
        """将原始输入转换为token IDs"""
        return [
            {
                "input_ids": torch.tensor(
                    [self.vocab.get(tag, self.vocab['[UNK]']) for tag in tags],
                    dtype=torch.long
                ),
                "target_ids": torch.tensor([self.vocab.get(tag, self.vocab['[UNK]']) for tag in tags],
                    dtype=torch.long
                ),
                "attention_mask": torch.ones(len(tags), dtype=torch.long)
            }
            for tags in input_tags_batch
        ]
    
    def calculate_distances(self, tags: List[str], embeddings: torch.Tensor) -> Dict[FrozenSet[str], float]:
        # 距离矩阵
        distance_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # 生成索引对
        n = len(tags)
        indices = itertools.product(range(n), repeat=2)
        
        # 生成键值对
        tag_pairs = (frozenset({tags[i], tags[j]}) for i, j in indices)
        distances_values = distance_matrix.flatten().tolist()
        
        return dict(zip(tag_pairs, distances_values))