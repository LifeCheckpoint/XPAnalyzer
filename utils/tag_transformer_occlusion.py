from config.configs import ExperimentConfig
from data.tag_dataset import collate_fn
from typing import List, Dict, Tuple
import itertools
import torch

cfg = ExperimentConfig()

class TagTransformerOcclusionAnalyzer:
    def __init__(self, model, vocab, device=cfg.device):
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.device = device
        self.id_to_vocab = {idx: token for token, idx in vocab.items()}

    def analyze_impact(
        self,
        input_tags: List[str],
        target_tag: str,
        max_combo_size: int = 3,
        occlusion_token: str = '[MASK]'
    ) -> Dict[Tuple[str], Dict[int, float]]:
        """
        遮挡分析，计算遮挡组合对目标标签概率的影响值。
        
        Args:
            input_tags: 输入标签序列（不包含[CLS]）
            target_tag: 要分析的目标标签
            max_combo_size: 最大遮挡组合大小
            occlusion_token: 遮挡时使用的特殊标记（默认为[MASK]）
            
        Returns:
            {遮挡组合: {位置: 影响值}}: Dict
        """
        if (target_id := self.vocab.get(target_tag)) is None:
            raise ValueError(f"目标标签 '{target_tag}' 不在词汇表中")
        
        # 输入序列构造与目标标签位置
        original_tokens, original_probs = self._get_original_probs(input_tags, target_id)
        positions = self._get_target_positions(original_tokens)
        
        return self._run_occlusion_analysis(
            original_tokens=original_tokens,
            original_probs=original_probs,
            target_id=target_id,
            positions=positions,
            max_combo_size=max_combo_size,
            occlusion_token=occlusion_token
        )

    def _get_original_probs(self, input_tags: List[str], target_id: int):
        """获取原始输入的概率分布"""
        # 构建输入序列
        token_ids = [self.vocab['[CLS]']] + [self.vocab.get(tag, self.vocab['[UNK]']) for tag in input_tags]
        
        with torch.no_grad():
            logits = self.model(*self._prepare_inputs([token_ids]))["logits"]
        
        # softmax 概率归一化输出
        probs = torch.softmax(logits[0, :len(token_ids)], dim=-1)
        return token_ids, probs[:, target_id].cpu()

    def _get_target_positions(self, token_ids: List[int]):
        """查找目标位置"""
        mask_id = self.vocab.get('[MASK]', -1)
        mask_positions = [i for i, tid in enumerate(token_ids) if tid == mask_id]
        
        return mask_positions if mask_positions else [
            i for i in range(1, len(token_ids))  # 跳过[CLS]
            if token_ids[i] != self.vocab['[PAD]']
        ]

    def _run_occlusion_analysis(
            self,
            original_tokens,
            original_probs,
            target_id, 
            positions,
            max_combo_size,
            occlusion_token
        ):
        """遮挡分析逻辑"""

        impact_results = {}
        valid_indices = [
            i for i in range(1, len(original_tokens)) if original_tokens[i] != self.vocab['[PAD]']
        ]
        
        occlusion_id = self.vocab[occlusion_token]
        
        for combo_size in range(1, max_combo_size + 1):
            for indices in itertools.combinations(valid_indices, combo_size):
                # 构建遮挡序列
                occluded_tokens = original_tokens.copy()
                for idx in indices:
                    occluded_tokens[idx] = occlusion_id
                
                # 获取遮挡后概率
                with torch.no_grad():
                    logits = self.model(*self._prepare_inputs([occluded_tokens]))["logits"]
                occluded_probs = torch.softmax(logits[0, :len(occluded_tokens)], dim=-1)[:, target_id]
                
                # 计算影响值
                impact = {pos: (original_probs[pos] - occluded_probs[pos]).item() for pos in positions}
                
                combo = tuple(self.id_to_vocab[original_tokens[i]] for i in indices)
                impact_results[combo] = impact
                
        return impact_results

    def _prepare_inputs(self, batch_token_ids):
        """输入预处理"""
        batch = [{
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'target_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.ones(len(ids), dtype=torch.long)
        } for ids in batch_token_ids]
        
        padded = collate_fn(batch, self.vocab['[PAD]'])
        return (
            padded['input_ids'].to(self.device),
            padded['attention_mask'].to(self.device)
        )