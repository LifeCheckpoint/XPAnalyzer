from config.configs import ExperimentConfig
import torch
from data.tag_dataset import collate_fn

cfg = ExperimentConfig()

class TagPredictor:
    def __init__(self, model, vocab, device=cfg.device):
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.device = device
        self.id_to_vocab = {idx: token for token, idx in vocab.items()}

    def predict(self, input_tags_batch, top_k=10):
        """
        ## 预测标签 Top K 信息

        input_tags_batch: List[List[str]]，不需要包含 [CLS]

        返回格式

        ```
        [
            [   # 序列1预测
                [ {"token": "[CLS]", "score": 0.99}, {...}, ... ],  # 位置0
                [ {"token": "傲娇", "score": 0.95}, {...}, ... ],   # 位置1
                [ {"token": "大小姐", "score": 0.8}, {...}, ... ]   # 位置2
            ],
            [   # 序列2预测
                [ {"token": "[CLS]", "score": 0.98}, {...}, ... ],  # 位置0
                [ {"token": "银发", "score": 0.96}, {...}, ... ],   # 位置1
                [ {"token": "萝莉", "score": 0.93}, {...}, ... ],   # 位置2
                [ {"token": "猫耳", "score": 0.91}, {...}, ... ]    # 位置3
            ]
        ]
        ```
        """

        batch_data = self.prepare_batch(input_tags_batch)
        padded_input = self.collate_batch(batch_data)
        logits = self._model_forward(padded_input)
        return self._process_outputs(logits, batch_data, top_k)

    def prepare_batch(self, input_tags_batch):
        """将原始输入转换为token IDs"""
        return [
            {
                "input_ids": torch.tensor(
                    [self.vocab['[CLS]']] + [self.vocab.get(tag, self.vocab['[UNK]']) for tag in tags],
                    dtype=torch.long
                ),
                "target_ids": torch.tensor([self.vocab.get(tag, self.vocab['[UNK]']) for tag in tags],
                    dtype=torch.long
                ),
                "attention_mask": torch.ones(len(tags)+1, dtype=torch.long) # +1 for [CLS]
            }
            for tags in input_tags_batch
        ]

    def collate_batch(self, batch_data):
        """数据长度对齐"""
        padded = collate_fn(batch_data, pad_token_id=self.vocab['[PAD]'])
        return {k: v.to(self.device) for k, v in padded.items()}

    def _model_forward(self, inputs):
        """推理"""
        with torch.no_grad():
            logits = self.model(inputs['input_ids'], inputs['attention_mask'])["logits"]
        return logits

    def _process_outputs(self, logits, batch_data, top_k):
        """处理模型输出，获取 Top K 预测"""
        top_k = max(1, min(top_k, len(self.vocab)))
        batch_results = []
        
        for logit, data in zip(logits, batch_data):
            seq_len = len(data['input_ids'])
            logit = logit[:seq_len]  # 移除padding
            
            # 获取整个序列的top-k预测
            values, indices = torch.topk(logit, top_k, dim=-1)  # [seq_len, top_k]
            
            # 转换为最简格式
            seq_predictions = []
            for pos in range(seq_len):
                pos_preds = [
                    {
                        "token": self.id_to_vocab.get(indices[pos][i].item(), '[UNK]'),
                        "score": values[pos][i].item()
                    }
                    for i in range(top_k)
                ]
                seq_predictions.append(pos_preds)
                
            batch_results.append(seq_predictions)
            
        return batch_results