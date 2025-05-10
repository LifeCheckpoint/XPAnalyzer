import json
import torch
import numpy as np
from typing import List, Dict, Union, FrozenSet
from dataclasses import dataclass
from config.configs import ExperimentConfig

cfg = ExperimentConfig()

@dataclass
class TagMark:
    tags: List[str]
    score: float

class TagMarker:
    def __init__(
        self,
        tags_distance: Dict[FrozenSet[str], float],
        full_embedding: Dict[str, torch.Tensor],
        weights_path: str = cfg.dataset.tags_weights_dataset_path,
    ):
        self._validate_embeddings(full_embedding)
        self.tags_distance = tags_distance
        self.full_embedding = full_embedding
        self.tags = list(full_embedding.keys())

        # 过滤特殊标签
        for tag in list(self.tags):
            if tag.startswith("[") and tag.endswith("]"):
                self.tags.remove(tag)
        
        self.min_distance, self.max_distance = self._compute_bounds()
        
        # 可选加载权重
        self.tags_weights = self._load_weights(weights_path) if weights_path else None
        self.avg_weight = sum(self.tags_weights.values()) / len(self.tags_weights) if self.tags_weights else 1.0

    def _validate_embeddings(self, embeddings: Dict[str, torch.Tensor]):
        """校验嵌入向量维度"""
        for tag, emb in embeddings.items():
            if emb.shape != (cfg.model.embed_dim,):
                raise ValueError(
                    f"Invalid embedding for tag '{tag}': "
                    f"expected shape [{cfg.model.embed_dim}], got {emb.shape}"
                )

    def _compute_bounds(self) -> tuple[float, float]:
        """计算有效距离范围（排除0距离）"""
        distances = [d for d in self.tags_distance.values() if d > cfg.eps]
        return min(distances), max(distances)

    def _load_weights(self, path: str) -> Dict[str, float]:
        """加载预定义标签权重"""
        with open(path, "r", encoding="utf-8") as f:
            weights = json.load(f)
            if not isinstance(weights, dict):
                raise ValueError(f"Invalid weights format: expected dict, got {type(weights)}")
            return weights

    def _normalize_distance(self, distance: float) -> float:
        """归一化距离到[0,1]范围并应用幂变换"""
        if distance < cfg.eps:
            return 0.0
            
        normalized = (distance - self.min_distance) / (self.max_distance - self.min_distance)
        return normalized ** cfg.mark.distance_transform_power

    def evaluate(self, marks: List[TagMark], threshold: int = 3) -> Dict[str, float]:
        """
        计算最终标签评分

        只计算出现过阈值次数以上的 Tag
        """
        final_scores = {tag: 0.0 for tag in self.tags}

        # 统计每个标签的出现次数
        tag_counts = {tag: 0 for tag in self.tags}
        for mark in marks:
            if mark["score"] == None:
                continue
            for tag in mark["tags"]:
                tag_counts[tag] += 1

        # 过滤掉出现次数少于阈值的标签
        tags_to_remove = [tag for tag, count in tag_counts.items() if count < threshold]
        for tag in tags_to_remove:
            for mark in marks:
                if tag in mark["tags"]:
                    mark["tags"].remove(tag)

        # 基础得分
        # 测试组（有该Tag的角色）的平均评分 avg1
        # 对照组（无该Tag的角色）的平均评分 avg2
        # 差异 delta = avg1 - avg2
        for tag in self.tags:
            test_group = [mark["score"] for mark in marks if tag in mark["tags"]]
            control_group = [mark["score"] for mark in marks if tag not in mark["tags"]]
            if len(test_group) == 0 or len(control_group) == 0:
                continue
            avg_test = sum(test_group) / len(test_group)
            avg_control = sum(control_group) / len(control_group)
            delta = avg_test - avg_control
            final_scores[tag] = delta * 10
        
        for mark in marks:
            if mark["score"] == None:
                continue

            # 计算当前标签组对所有标签的贡献
            group_impact = self._compute_group_impact(mark["tags"])
            for tag, impact in group_impact.items():
                final_scores[tag] += impact * mark["score"] * cfg.mark.distance_transform_scale

        # 置信度因子
        # n_test 为具有该 tag 的组数，n_control 为没有该 tag 的组数
        countFactor = lambda n_test, n_control: min(1.8, max(0, np.log((n_test * n_control) / (n_test + n_control) / 2 + 1) - 0.7))
        for tag in final_scores:
            n_test = sum(1 for mark in marks if tag in mark["tags"])
            n_control = len(marks) - n_test
            final_scores[tag] *= countFactor(n_test, n_control)
        
        # 预定义权重
        if self.tags_weights:
            for tag in final_scores:
                final_scores[tag] *= self.tags_weights.get(tag, self.avg_weight) ** cfg.mark.pre_weights_transform_power
        
        return final_scores

    def _compute_group_impact(self, group_tags: List[str]) -> Dict[str, float]:
        """计算单个标签组的影响"""
        impact = {}
        for target_tag in self.tags:
            # 计算组内标签到目标标签的标准距离
            total = 0.0
            for group_tag in group_tags:
                distance = self.tags_distance.get(
                    frozenset({group_tag, target_tag}),
                    float(self.max_distance)  # 如果不存在
                )
                total += self._normalize_distance(distance)
            impact[target_tag] = total / len(group_tags)
        return impact