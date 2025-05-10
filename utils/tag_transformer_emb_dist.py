from config.configs import ExperimentConfig
from data.tag_dataset import collate_fn
from typing import List, Dict, FrozenSet
import itertools
import torch

cfg = ExperimentConfig()

class TagTransformerEmbeddingDistance:
    """
    统计计算标签嵌入距离
    """
    def __init__(self, model, vocab, device=cfg.device):
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.device = device
        self.id_to_vocab = {idx: token for token, idx in vocab.items()}

    def compute_embedding_distance(self, tags: List[str]) -> Dict[FrozenSet[str], float]:
        """
        计算标签嵌入距离
        """
        # 将标签转换为索引
        tags_batch = self.prepare_batch([[tag] for tag in tags])
        padded_input = collate_fn(tags_batch, pad_token_id=self.vocab['[PAD]'])

        padded_input = {k: v.to(self.device) for k, v in padded_input.items()}

        # 获取标签嵌入
        with torch.no_grad():
            embeddings = self.model(padded_input["input_ids"], padded_input["attention_mask"])["encoded"]

        # 计算嵌入距离
        distances = {}
        with torch.no_grad():
            embeddings = torch.reshape(embeddings, (-1, cfg.model.embed_dim))
            distances = self.calculate_distances_optimized(tags, embeddings)
        
        return distances
    
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
                "attention_mask": torch.ones(len(tags), dtype=torch.long) # +1 for [CLS]
            }
            for tags in input_tags_batch
        ]
    
    def calculate_distances_optimized(self, tags, embeddings):
        embeddings = embeddings.to(self.device)
        
        # 计算距离矩阵
        diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)
        distance_matrix = torch.norm(diff, dim=2)
        
        # 生成索引和标签对
        n = len(tags)
        indices = list(itertools.combinations(range(n), 2))
        tag_pairs = [frozenset({tags[i], tags[j]}) for i, j in indices]
        
        # 提取距离值
        rows = torch.tensor([i for i, j in indices], device=embeddings.device)
        cols = torch.tensor([j for i, j in indices], device=embeddings.device)
        distances_values = distance_matrix[rows, cols].cpu().tolist()
        
        return dict(zip(tag_pairs, distances_values))