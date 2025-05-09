from config.configs import ExperimentConfig
from torch import nn, optim
from torch.nn import functional as F
import torch

cfg = ExperimentConfig()

class TagTransformerTrainer:
    def __init__(self, model, vocab, device=cfg.device):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab['[PAD]'])
        self.optimizer = optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)

    def train_epoch(self, dataloader):
        """MASK 预测 + 对比学习训练"""
        self.model.train()
        total_loss_critetion = 0
        total_loss_contrastive = 0

        for batch in dataloader:
            inputs = batch['input_ids'].to(self.device)
            targets = batch['target_ids'].to(self.device)
            mask = batch['attention_mask'].to(self.device)

            self.optimizer.zero_grad()

            model_output = self.model(inputs, mask)
            logits = model_output['logits']
            contrastive_output = model_output['sentence_emb']

            # 交叉熵
            loss_criterion = self.criterion(
                logits.view(-1, len(self.vocab)),
                targets.view(-1)
            )
            # 对比损失
            loss_contrastive = self.compute_contrastive_loss(contrastive_output)
            # 总损失
            loss = cfg.train.mask_predict_loss_weight * loss_criterion + cfg.train.contrastive_loss_weight * loss_contrastive

            loss.backward()
            self.optimizer.step()

            total_loss_critetion += loss_criterion.item() * cfg.train.mask_predict_loss_weight
            total_loss_contrastive += loss_contrastive.item() * cfg.train.contrastive_loss_weight

        return {
            "critetion_loss": total_loss_critetion / len(dataloader),
            "contrastive_loss": total_loss_contrastive / len(dataloader),
        }
    
    def compute_contrastive_loss(self, sentence_emb):
        # L2归一化
        sentence_emb = F.normalize(sentence_emb, p=2, dim=1)
        
        # 相似度矩阵
        sim_matrix = torch.matmul(sentence_emb, sentence_emb.T)  # (batch_size, batch_size)
        sim_matrix /= cfg.train.contrastive_temperature

        labels = torch.arange(sentence_emb.size(0), device=self.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss