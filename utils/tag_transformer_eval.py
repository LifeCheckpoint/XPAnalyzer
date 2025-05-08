from configs import EvaluateConfig
import torch
from torch import nn

testCFG = EvaluateConfig()

class TagTransformerEvaluator:
    def __init__(self, model, vocab, device=testCFG.device):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab['[PAD]'])

    def evaluate(self, dataloader):
        """计算平均损失和Top-1/Top-5准确率"""
        self.model.eval()
        total_loss = 0
        correct_top1 = 0
        correct_top5 = 0
        total_elements = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input_ids'].to(self.device)
                targets = batch['target_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)

                logits, _ = self.model(inputs, mask)
                loss = self.criterion(logits.view(-1, len(self.vocab)), targets.view(-1))
                total_loss += loss.item()

                # 计算准确率
                valid_targets = targets[mask == 1]
                valid_logits = logits[mask == 1]
                total_elements += valid_targets.numel()

                # Top-1 准确率
                predicted_top1 = torch.argmax(valid_logits, dim=-1)
                correct_top1 += (predicted_top1 == valid_targets).sum().item()

                # Top-5 准确率
                k = min(5, len(self.vocab))
                _, predicted_top5 = torch.topk(valid_logits, k, dim=-1)
                correct_top5 += torch.any(predicted_top5 == valid_targets.unsqueeze(1), dim=-1).sum().item()

        return {
            "loss": total_loss / len(dataloader),
            "top1_acc": correct_top1 / total_elements,
            "top5_acc": correct_top5 / total_elements
        }
    
    def get_embeddings(self, inputs, attention_mask):
        self.model.eval()
        with torch.no_grad():
            _, embeddings = self.model(inputs, attention_mask)
        return embeddings